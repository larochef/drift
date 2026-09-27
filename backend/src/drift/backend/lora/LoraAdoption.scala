package drift.backend.lora

import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.*
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.typesafe.scalalogging.Logger

/** Taking in a folder of the LoRA store that no entity claims
  * (`specs/33-lora-sources.md`): one left behind by a schema break, or copied
  * in by hand. Its sidecar and its weight files are enough to rebuild the
  * entity, and a half-transferred file is simply queued again.
  *
  * The opposite way round from an install: the files are there first and the
  * entity is made to match them.
  */
final private[lora] class LoraAdoption(
    storage: StorageService,
    downloads: LoraDownloads,
    lorasRoot: Path
) {

  private val logger = Logger[LoraAdoption]

  /** Adopts an orphan folder of the LoRA store — one no entity references, left
    * behind by a schema break or copied in by hand — rebuilding the entity for
    * `request.architectureId` from the sidecar and the weight files, and moving
    * the folder under that architecture. A `.part` leftover counts as a weight
    * file and its transfer resumes.
    */
  def adopt(request: AdoptLoraRequest): InstallLoraResponse = {
    if (!storage.exists("architectures", request.architectureId))
      return InstallLoraResponse(error =
        Some(s"architecture '${request.architectureId}' does not exist")
      )
    val provided = Paths.get(request.path).toAbsolutePath.normalize
    val root = lorasRoot.toAbsolutePath.normalize
    if (!provided.startsWith(root) || provided == root)
      return InstallLoraResponse(error =
        Some(s"'$provided' is not inside the LoRA store")
      )
    val folder =
      if (Files.isDirectory(provided)) provided else provided.getParent
    if (folder == null || !Files.isDirectory(folder) || folder == root)
      return InstallLoraResponse(error =
        Some(s"'$provided' has no adoptable folder")
      )

    val sidecar = readSidecar(folder)
    val folderName = folder.getFileName.toString
    val label = sidecar.map(_.name).filter(_.nonEmpty).getOrElse(folderName)
    // Folder names have always ended in -<civitaiModelId>; the sidecar wins.
    val civitaiModelId = sidecar
      .map(_.modelId)
      .filter(_.nonEmpty)
      .orElse(folderName.split('-').lastOption.filter(_.forall(_.isDigit)))
      .getOrElse("")
    val versionId = sidecar.map(_.versionId).getOrElse("")
    val versionName = sidecar.map(_.versionName).getOrElse("")

    val weightNames = regularFilesIn(folder)
      .map(_.getFileName.toString)
      .map(_.stripSuffix(".part"))
      .filter(LoraManager.isWeightFile)
      .distinct
    if (weightNames.isEmpty)
      return InstallLoraResponse(error =
        Some(s"'$folderName' holds no weight files to adopt")
      )

    val id = LoraManager.idOf(request.architectureId, label, civitaiModelId)
    if (storage.exists("loras", id))
      return InstallLoraResponse(error =
        Some(
          s"LoRA '$id' already exists for this architecture — delete it " +
            "(or this folder) instead of adopting"
        )
      )
    val nsfw =
      Option(folder.getParent).exists(_.getFileName.toString == "nsfw") ||
        sidecar.flatMap(_.nsfwLevel).exists(_ >= 8)
    val placed = Lora(
      id = id,
      architectureId = request.architectureId,
      label = label,
      nsfw = nsfw,
      triggerWords = sidecar.map(_.trainedWords).getOrElse(Nil),
      tags = sidecar.map(_.tags).getOrElse(Nil),
      description = sidecar.flatMap(_.description).map(_.take(2000)),
      createdAt = System.currentTimeMillis()
    )
    val target = lorasRoot.resolve(placed.folderRelativePath)
    val files = weightNames.map { diskName =>
      // A Civitai install stores `<fileId>-<filename>`; with a Civitai model
      // to attach it to, that is where the file came from. Anything else was
      // copied in by hand and is its own source: kept under its own name.
      val source = diskName.split("-", 2) match {
        case Array(fileId, filename)
            if civitaiModelId.nonEmpty && fileId.nonEmpty &&
              fileId.forall(_.isDigit) && filename.nonEmpty =>
          Civitai(civitaiModelId, versionId, fileId, filename)
        case _ => Local(target.resolve(diskName).toString)
      }
      val onDisk = folder.resolve(diskName)
      LoraFile(
        fileName = diskName,
        source = source,
        stage = LoraManager.stageOf(diskName, versionName),
        sizeBytes =
          if (Files.isRegularFile(onDisk)) Some(Files.size(onDisk)) else None
      )
    }
    val lora = placed.copy(files = files)
    if (folder != target)
      try {
        Files.createDirectories(target.getParent)
        Files.move(folder, target, StandardCopyOption.ATOMIC_MOVE)
      } catch {
        case NonFatal(err) =>
          return InstallLoraResponse(error =
            Some(s"could not move '$folder' to '$target': ${err.getMessage}")
          )
      }
    storage.save("loras", lora.id, lora)
    // Files on disk get their completed job; a `.part` leftover resumes —
    // with no sidecar there is nothing to fetch from, and that fails loudly.
    lora.files.foreach(file => downloads.queueDownload(lora, file))
    InstallLoraResponse(lora = Some(lora))
  }

  private def readSidecar(folder: Path): Option[ModelMetadata] =
    try {
      val sidecar = folder.resolve("drift-lora.json")
      if (!Files.isRegularFile(sidecar)) None
      else Some(readFromString[ModelMetadata](Files.readString(sidecar)))
    } catch { case NonFatal(_) => None }

  private def regularFilesIn(folder: Path): List[Path] = {
    val stream = Files.list(folder)
    try stream.iterator().asScala.filter(Files.isRegularFile(_)).toList
    finally stream.close()
  }

  // ------------------------------------------------------------------ update
}
