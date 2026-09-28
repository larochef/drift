package drift.backend.lora

import drift.backend.cache.ModelCache
import drift.backend.download.*
import drift.backend.routes.CivitaiClient
import drift.backend.storage.StorageService
import drift.shared.*

import java.net.http.*
import java.nio.file.*
import java.util.Comparator
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** Owns the LoRA store (`specs/09-lora-management.md`): one folder per LoRA
  * under `~/.cache/drift/loras/<architectureId>/<sfw|nsfw>/<id>/`, holding the
  * weight file(s) — a wan 2.2 high/low-noise pair is two files in the same
  * folder — and, for a Civitai install, the metadata sidecar and previews.
  * Files come from Civitai, a HuggingFace repository or the drift host, and the
  * official LoRAs of `reference/loras.json` install the same way
  * (`specs/33-lora-sources.md`).
  *
  * The user's tuning lives in the `Lora` entity (`~/.config/drift/loras/`),
  * never in the cache: the cache must stay safe to delete. Transfers share
  * [[Downloader]] (resume, sha256, progress) with model downloads but not
  * `DownloadManager`, which is keyed by registered `Model`s.
  */
final class LoraManager(
    storage: StorageService,
    civitaiClient: CivitaiClient,
    huggingFace: HuggingFaceDownloads,
    modelScope: drift.backend.modelscope.ModelScopeDownloads,
    downloader: Downloader,
    client: HttpClient,
    /** Resolved per request, so a token saved in Settings applies without a
      * restart.
      */
    civitaiToken: () => Option[String],
    /** The LoRAs drift offers to install, read once at start. */
    val catalog: List[Lora],
    val lorasRoot: Path
) {
  private val logger = Logger[LoraManager]

  /** What a LoRA folder carries beside its weights (`LoraSidecars`). */
  private val sidecars = LoraSidecars(client)

  /** Turning what a request ticked into LoRAs on disk (`LoraInstalls`). */
  private lazy val installs = LoraInstalls(
    storage,
    civitaiClient,
    huggingFace,
    modelScope,
    downloads,
    sidecars,
    catalog,
    lorasRoot
  )

  /** Taking in an orphan folder of the store (`LoraAdoption`). */
  private lazy val adoption = LoraAdoption(storage, downloads, lorasRoot)

  def adopt(request: AdoptLoraRequest): InstallLoraResponse =
    adoption.adopt(request)

  /** Getting the weights onto the disk (`LoraDownloads`). */
  private val downloads = LoraDownloads(
    storage,
    huggingFace,
    modelScope,
    downloader,
    civitaiToken,
    lorasRoot
  )

  def listJobs: List[LoraDownloadJob] = downloads.listJobs

  def cancelDownload(
      loraId: String,
      fileName: String
  ): Option[LoraDownloadJob] =
    downloads.cancel(loraId, fileName)

  /** Re-queues every installed LoRA file that is not on disk, once at start
    * (`LoraDownloads`).
    */
  def resumeInterrupted(): Unit = downloads.resumeInterrupted()

  /** Every installed LoRA entity. */
  def list: List[Lora] = storage.list[Lora]("loras")

  /** Installs what a request ticked, from whichever site or folder it names
    * (`LoraInstalls`).
    */
  def install(request: InstallLoraRequest): InstallLoraResponse =
    installs.install(request)

  /** Saves the user-editable fields; a flipped `nsfw` physically moves the
    * folder between sfw/ and nsfw/. Identity fields stay as stored.
    */
  def update(id: String, incoming: Lora): Option[Lora] =
    storage.get[Lora]("loras", id).map { existing =>
      // Where the files come from is not the user's to edit; their stage is.
      val files = existing.files.map(file =>
        incoming.files
          .find(_.fileName == file.fileName)
          .fold(file)(edited => file.copy(stage = edited.stage))
      )
      val updated = incoming.copy(
        id = existing.id,
        architectureId = existing.architectureId,
        files = files,
        createdAt = existing.createdAt
      )
      if (existing.nsfw != updated.nsfw) moveFolder(existing, updated)
      storage.save("loras", id, updated)
    }

  /** Moves the cache folder when the entity's placement changes — an nsfw flip,
    * or an install claiming the LoRA for another architecture.
    */
  private def moveFolder(from: Lora, to: Lora): Unit = {
    val source = lorasRoot.resolve(from.folderRelativePath)
    val target = lorasRoot.resolve(to.folderRelativePath)
    try
      if (Files.isDirectory(source)) {
        Files.createDirectories(target.getParent)
        Files.move(source, target, StandardCopyOption.ATOMIC_MOVE)
      }
    catch {
      case NonFatal(err) =>
        logger.warn(
          s"Moving LoRA '${to.id}' from '$source' to '$target' failed: " +
            err.getMessage
        )
    }
  }

  // ------------------------------------------------------------------ delete

  /** Deletes the entity and the cache folder — both placements, in case the
    * entity and the disk disagree about nsfw.
    */
  /** Makes one LoRA of two (`specs/33-lora-sources.md`): the other's files move
    * into this one's folder and the other entity goes. A wan 2.2 pair published
    * as two repositories arrives as two LoRAs, one stage each, and this is what
    * makes it the pair it is (François, 2026-09-18).
    *
    * The label is the user's: neither half's name says what the whole is. The
    * id does not change — it names the folder, and the files have just been
    * moved into it.
    */
  def pair(request: PairLoraRequest): InstallLoraResponse = {
    val kept = storage.get[Lora]("loras", request.loraId)
    val other = storage.get[Lora]("loras", request.otherId)
    (kept, other) match {
      case (None, _) =>
        LoraManager.failure(s"LoRA '${request.loraId}' no longer exists")
      case (_, None) =>
        LoraManager.failure(s"LoRA '${request.otherId}' no longer exists")
      case (Some(one), Some(two)) if one.id == two.id =>
        LoraManager.failure("a LoRA cannot be paired with itself")
      case (Some(one), Some(two)) if one.architectureId != two.architectureId =>
        LoraManager.failure(s"'${two.label}' is a LoRA of another architecture")
      case (Some(one), Some(two)) =>
        val taken = one.files.map(_.fileName).toSet
        val moved = two.files.map { file =>
          // Two repositories name their file the same way more often than
          // not (`adapter_model.safetensors`): the stage tells them apart.
          val name =
            if (taken.contains(file.fileName))
              s"${stageSlug(file.stage)}-${file.fileName}"
            else file.fileName
          moveFile(two, file, one, name)
          file.copy(fileName = name)
        }
        val paired = one.copy(
          label =
            Option(request.label.trim).filter(_.nonEmpty).getOrElse(one.label),
          files = one.files ++ moved,
          triggerWords = (one.triggerWords ++ two.triggerWords).distinct,
          tags = (one.tags ++ two.tags).distinct,
          description = one.description.orElse(two.description)
        )
        storage.save("loras", paired.id, paired)
        delete(two.id)
        // Whatever had not been fetched yet is fetched into its new place.
        paired.files.foreach(file => downloads.queueDownload(paired, file))
        InstallLoraResponse(lora = Some(paired))
    }
  }

  private def stageSlug(stage: LoraFileStage): String = stage match {
    case LoraFileStage.HighNoise => "high"
    case LoraFileStage.LowNoise  => "low"
    case LoraFileStage.General   => "other"
  }

  /** One file from one LoRA's folder to another's, under a name free there. A
    * file still downloading is left to its job: the entity says where it goes
    * now, and the resume fetches it there.
    */
  private def moveFile(
      from: Lora,
      file: LoraFile,
      to: Lora,
      name: String
  ): Unit = {
    val source = lorasRoot.resolve(from.storagePathOf(file))
    val target =
      lorasRoot.resolve(to.folderRelativePath).resolve(name)
    try
      if (Files.isRegularFile(source)) {
        Files.createDirectories(target.getParent)
        Files.move(source, target, StandardCopyOption.REPLACE_EXISTING)
      }
    catch {
      case NonFatal(err) =>
        logger.warn(
          s"Moving '$source' to '$target' while pairing failed: " +
            err.getMessage
        )
    }
  }

  def delete(id: String): Boolean =
    storage.get[Lora]("loras", id) match {
      case None       => storage.delete("loras", id)
      case Some(lora) =>
        List(lora.nsfw, !lora.nsfw)
          .map(flag =>
            lorasRoot.resolve(lora.copy(nsfw = flag).folderRelativePath)
          )
          .foreach(deleteRecursively)
        storage.delete("loras", id)
    }

  private def deleteRecursively(directory: Path): Unit =
    if (Files.isDirectory(directory))
      try {
        val stream = Files.walk(directory)
        try
          stream
            .sorted(Comparator.reverseOrder[Path]())
            .iterator()
            .asScala
            .foreach(Files.deleteIfExists)
        finally stream.close()
      } catch {
        case NonFatal(err) =>
          logger.warn(s"Deleting '$directory' failed: ${err.getMessage}")
      }

  // ----------------------------------------------------------------- sidecar
}

object LoraManager {

  /** A refusal, said the way every install answers — the reason and nothing
    * installed.
    */
  private[lora] def failure(reason: String): InstallLoraResponse =
    InstallLoraResponse(error = Some(reason))

  private val WeightExtensions =
    Set(".safetensors", ".ckpt", ".pt", ".gguf")

  def isWeightFile(name: String): Boolean =
    WeightExtensions.exists(name.toLowerCase.endsWith)

  /** A file's path without its extension, folders kept: a HuggingFace
    * repository's `high_noise_model` means little without its folder.
    */
  def labelOf(filename: String): String =
    filename.lastIndexOf('.') match {
      case -1  => filename
      case dot => filename.take(dot)
    }

  /** `<architectureId>-<slug>-<source>`, which also names the folder. */
  def idOf(architectureId: String, label: String, sourcePart: String): String =
    List(
      architectureId,
      Some(ModelCache.slug(label)).filter(_.nonEmpty).getOrElse("lora"),
      sourcePart
    ).filter(_.nonEmpty).mkString("-")

  private val HighNoisePattern = "(?i)high[ ._-]?noise".r
  private val LowNoisePattern = "(?i)low[ ._-]?noise".r

  /** The same thing said in fewer letters, as the repository sites tend to:
    * `..._HIGH.safetensors`, `wan22_hn_rank64.gguf`, a `high/` folder. A word
    * on its own only — "shallow" is not a low-noise LoRA, "highres" is not a
    * high-noise one.
    */
  private val HighPattern = "(?i)(?<![a-z0-9])(high|hn)(?![a-z0-9])".r
  private val LowPattern = "(?i)(?<![a-z0-9])(low|ln)(?![a-z0-9])".r

  /** Which stage a file belongs to, guessed from the names it goes by — its own
    * (with the folders above it, which are often where a repository says it),
    * its Civitai version's, its repository's. wan 2.2 pairs are marked in one
    * or another; correctable in the UI when the guess is wrong.
    *
    * "high noise" and "low noise" are looked for everywhere first: a file
    * called `high_noise` in a repository whose name says `low` is a high-noise
    * file. Only then does the short form count (François, 2026-09-18: Civitai
    * guesses well, the other sites should too).
    */
  def stageOf(names: String*): LoraFileStage = {
    def firstMatch(
        high: scala.util.matching.Regex,
        low: scala.util.matching.Regex
    ): Option[LoraFileStage] =
      names.iterator
        .flatMap(name =>
          if (high.findFirstIn(name).isDefined) Some(LoraFileStage.HighNoise)
          else if (low.findFirstIn(name).isDefined) Some(LoraFileStage.LowNoise)
          else None
        )
        .nextOption()

    firstMatch(HighNoisePattern, LowNoisePattern)
      .orElse(firstMatch(HighPattern, LowPattern))
      .getOrElse(LoraFileStage.General)
  }
}
