package drift.backend.conversion

import drift.backend.cache.*
import drift.backend.runtime.RuntimeManager
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.*
import scala.jdk.StreamConverters.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** Model conversion (`specs/25-model-conversion.md`): a cached weight file
  * quantized to GGUF by sd-cpp's own converter, written under
  * `models/<family>/converted/` of the drift cache — where the inventory
  * already lists it as a local file of that family — and registered as a model
  * of the family with the source model's parameters, so every run configuration
  * slot wanting the family can take it.
  *
  * Refusals happen here, before anything is queued: a source outside the cache,
  * a ComfyUI int8 file (sd-cpp writes its int8 tensors back unchanged), an
  * output that exists, too little disk, no sd-cpp runtime.
  */
final class ConversionManager(
    storage: StorageService,
    cache: ModelCache,
    runtimeManager: RuntimeManager,
    logsRoot: Path
) {
  private val logger = Logger[ConversionManager]
  private val jobs = ConversionJobs(logsRoot)

  sweepPartials()

  def listJobs: List[ConversionJob] = jobs.list

  def cancel(id: String): Boolean = jobs.cancel(id)

  /** What a cached file holds, for the modal's dtype line and size estimate. */
  def inspect(rawPath: String): Either[String, ModelFileInfo] =
    sourceFile(rawPath).flatMap(ModelFileInspector.inspect)

  def start(request: ConversionRequest): Either[String, ConversionJob] =
    for {
      source <- sourceFile(request.path)
      _ <- validated(request)
      info <- ModelFileInspector.inspect(source)
      preStep <- preStepFor(source, info)
      output <- outputFor(request)
      intermediate <- intermediateFor(output, preStep)
      _ <- enoughDisk(
        source,
        info,
        request.targetType,
        preStep.map(_.outputBytes).getOrElse(0L)
      )
      launch <- runtimeManager.resolveForLaunch(
        RuntimeTool.SdCpp,
        // sd-cli converts; only sd-cpp ships one
        RuntimeEngine.SdCpp,
        None
      )
      cli <- RuntimeManager.sdCliOf(launch)
    } yield {
      val sourceModel = modelsOf(source).headOption
      val partFile = output.resolveSibling(s"${output.getFileName}.part")
      val job = ConversionJob(
        id = jobs.nextId(),
        sourcePath = source.toString,
        sourceLabel =
          sourceModel.map(_.label).getOrElse(source.getFileName.toString),
        outputPath = output.toString,
        familyId = request.familyId,
        targetType = request.targetType,
        rules = request.rules.trim,
        state = ConversionState.Queued,
        detail = "queued",
        startedAt = System.currentTimeMillis()
      )
      Files.createDirectories(output.getParent)
      logger.info(
        s"Conversion ${job.id} queued: ${source.getFileName} → ${output.getFileName} (${request.targetType})"
      )
      jobs.enqueue(
        job,
        QueuedConversion(
          id = job.id,
          partFile = partFile,
          command = SdCppConvert.command(
            cli,
            intermediate.getOrElse(source),
            partFile,
            request
          ),
          launch = launch,
          complete = register(_, partFile, output, sourceModel),
          dequantize = intermediate.map(path =>
            Dequantization(source, path, request.keepIntermediate)
          )
        )
      )
    }

  // ------------------------------------------------------------ refusals

  private def sourceFile(rawPath: String): Either[String, Path] = {
    val path = Path.of(rawPath).toAbsolutePath.normalize
    val roots = (cache.driftRoot :: cache.huggingFaceRoot :: cache.extraRoots)
      .map(_.toAbsolutePath.normalize)
    if (!roots.exists(path.startsWith))
      Left("Only files of the model cache can be converted.")
    else if (!Files.isRegularFile(path))
      Left(s"'${path.getFileName}' is not a file.")
    else Right(path)
  }

  private val FamilyName = "^[A-Za-z0-9][A-Za-z0-9._-]*$".r
  private val OutputName = "^[A-Za-z0-9][A-Za-z0-9._-]*\\.gguf$".r
  private val TypeName = "^[A-Za-z0-9_]+$".r

  private def validated(request: ConversionRequest): Either[String, Unit] =
    if (!FamilyName.matches(request.familyId) || request.familyId == "..")
      Left("The family must be a plain name: letters, digits, '.', '_', '-'.")
    else if (
      !OutputName.matches(request.outputName) || request.outputName == ".."
    )
      Left("The output name must be a plain file name ending in .gguf.")
    else if (!TypeName.matches(request.targetType))
      Left("The type must be a sd-cpp type name, such as q8_0 or q4_K.")
    else if (request.rules.exists(c => c == '\n' || c == '\r'))
      Left("Tensor type rules are one comma-separated line.")
    else if (request.threads.exists(_ < 1))
      Left("Threads must be at least 1.")
    else Right(())

  /** sd-cpp's `tensor_should_be_converted` returns false for every ComfyUI int8
    * tensor, and its converter never applies fp8 scales (upstream's Ideogram 4
    * recipe rescales with a script first) — such files go through the
    * [[Dequantizer]] before sd-cli sees them. The plan is made here so a file
    * drift cannot dequantize is refused before anything is queued.
    */
  private def preStepFor(
      source: Path,
      info: ModelFileInfo
  ): Either[String, Option[Dequantizer.Plan]] =
    if (info.comfyQuant.isDefined || info.scaledFp8)
      Dequantizer.plan(source).map(Some(_))
    else Right(None)

  /** `<output base>-dequantized.safetensors` beside the output, when the source
    * needs the pre-step.
    */
  private def intermediateFor(
      output: Path,
      preStep: Option[Dequantizer.Plan]
  ): Either[String, Option[Path]] = preStep match {
    case None    => Right(None)
    case Some(_) =>
      val name = output.getFileName.toString.stripSuffix(".gguf")
      val intermediate = output.resolveSibling(s"$name-dequantized.safetensors")
      val part = output.resolveSibling(s"${intermediate.getFileName}.part")
      if (Files.exists(intermediate))
        Left(s"'${intermediate.getFileName}' already exists.")
      else if (Files.exists(part))
        Left(
          s"'${intermediate.getFileName}' is being written by another conversion."
        )
      else Right(Some(intermediate))
  }

  private def outputFor(request: ConversionRequest): Either[String, Path] = {
    val output = cache.driftRoot
      .resolve("models")
      .resolve(request.familyId)
      .resolve(ConversionManager.ConvertedDirectory)
      .resolve(request.outputName)
      .toAbsolutePath
      .normalize
    val part = output.resolveSibling(s"${output.getFileName}.part")
    if (Files.exists(output)) Left(s"'${request.outputName}' already exists.")
    else if (Files.exists(part))
      Left(s"'${request.outputName}' is being written by another conversion.")
    else Right(output)
  }

  private def enoughDisk(
      source: Path,
      info: ModelFileInfo,
      targetType: String,
      intermediateBytes: Long
  ): Either[String, Unit] =
    try {
      val models = cache.driftRoot.resolve("models")
      Files.createDirectories(models)
      val usable = Files.getFileStore(models).getUsableSpace
      val estimate = ConversionTypes
        .estimateBytes(info.parameterCount, targetType)
        .getOrElse(Files.size(source))
      val needed = ((estimate + intermediateBytes) * 1.1).toLong
      Either.cond(
        usable >= needed,
        (),
        s"Not enough disk: about ${LaunchBlocker.humanBytes(needed)} needed, ${LaunchBlocker
            .humanBytes(usable)} free under ${cache.driftRoot}."
      )
    } catch {
      case NonFatal(err) =>
        Left(s"Disk space could not be checked: ${err.getMessage}")
    }

  /** The registered models whose weights are this very file. */
  private def modelsOf(source: Path): List[Model] =
    storage.list[Model]("models").filter { model =>
      cache.resolve(model.source) match {
        case CacheEntry.Present(path, _) =>
          try Files.isSameFile(path, source)
          catch { case NonFatal(_) => false }
        case _ => false
      }
    }

  // -------------------------------------------------------- registration

  /** Runs on the worker once sd-cli exited cleanly: the output is moved into
    * place, checked to be a GGUF with tensors in it, and registered as a model
    * of the family — the source model's id and label with the type appended,
    * its parameters carried over.
    */
  private def register(
      job: ConversionJob,
      partFile: Path,
      output: Path,
      sourceModel: Option[Model]
  ): Either[String, ConversionJob] =
    try {
      Files.move(partFile, output, StandardCopyOption.ATOMIC_MOVE)
      ModelFileInspector.inspect(output) match {
        case Left(reason) =>
          Files.deleteIfExists(output)
          Left(s"sd-cli's output is not a GGUF: $reason")
        case Right(info) if info.tensorCount == 0 =>
          Files.deleteIfExists(output)
          Left("sd-cli wrote a GGUF with no tensors in it.")
        case Right(_) =>
          val taken = storage.list[Model]("models").map(_.id).toSet
          val fileBase = output.getFileName.toString.stripSuffix(".gguf")
          val base = sourceModel
            .map(_.id)
            .getOrElse(ConversionManager.slug(fileBase))
          val typeSlug = job.targetType.toLowerCase.replace('_', '-')
          val wanted =
            if (base.endsWith(s"-$typeSlug")) base else s"$base-$typeSlug"
          val id =
            if (!taken(wanted)) wanted
            else
              Iterator.from(2).map(n => s"$wanted-$n").filterNot(taken).next()
          val model = Model(
            id = id,
            familyId = job.familyId,
            label =
              s"${sourceModel.map(_.label).getOrElse(fileBase)} · ${job.targetType.toUpperCase}",
            source = Local(output.toString),
            format = "gguf",
            parameters = sourceModel.map(_.parameters).getOrElse(Map.empty),
            removedParameters =
              sourceModel.map(_.removedParameters).getOrElse(Nil)
          )
          storage.save[Model]("models", id, model)
          Right(job.copy(modelId = Some(id)))
      }
    } catch {
      case NonFatal(err) =>
        Left(s"registering the converted model failed: ${err.getMessage}")
    }

  /** A `.part` under a `converted/` directory is a conversion this process
    * never finished — the downloads' own `.part` files elsewhere under
    * `models/` are resumable and left alone.
    */
  private def sweepPartials(): Unit = {
    val models = cache.driftRoot.resolve("models")
    if (Files.isDirectory(models))
      try {
        val stale = Files
          .walk(models, 3)
          .toScala(List)
          .filter(path =>
            Files.isRegularFile(path) &&
              path.getFileName.toString.endsWith(".part") &&
              path.getParent.getFileName.toString == ConversionManager.ConvertedDirectory
          )
        stale.foreach { path =>
          logger.info(s"Removing unfinished conversion output $path")
          Files.deleteIfExists(path)
        }
      } catch {
        case NonFatal(err) =>
          logger.warn(
            s"Sweeping unfinished conversions failed: ${err.getMessage}"
          )
      }
  }
}

object ConversionManager {

  /** Where a family's conversions land: `models/<family>/converted/`. */
  val ConvertedDirectory: String = "converted"

  def slug(text: String): String =
    text.toLowerCase
      .map(c => if (c.isLetterOrDigit) c else '-')
      .split('-')
      .filter(_.nonEmpty)
      .mkString("-") match {
      case ""    => "model"
      case other => other
    }
}
