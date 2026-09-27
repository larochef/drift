package drift.backend.lora

import drift.backend.cache.ModelCache
import drift.backend.download.HuggingFaceDownloads
import drift.backend.routes.CivitaiClient
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.*

/** Turning what a user ticked into LoRAs on disk
  * (`specs/09-lora-management.md`, `specs/33-lora-sources.md`): the files of a
  * Civitai model, of a HuggingFace or ModelScope repository, of a folder on
  * this host, or of drift's own catalog — each looked up on its site, refused
  * by name when it cannot be a LoRA of that architecture, and saved as the
  * entities their grouping asks for.
  *
  * The entity is written before its weights arrive; `LoraDownloads` fetches
  * them afterwards.
  */
final private[lora] class LoraInstalls(
    storage: StorageService,
    civitaiClient: CivitaiClient,
    huggingFace: HuggingFaceDownloads,
    modelScope: drift.backend.modelscope.ModelScopeDownloads,
    downloads: LoraDownloads,
    sidecars: LoraSidecars,
    catalog: List[Lora],
    lorasRoot: Path
) {

  /** Which entity a group of fetched files lands in (`LoraPlacement`). */
  private val placement = LoraPlacement(storage, downloads, lorasRoot)

  def install(request: InstallLoraRequest): InstallLoraResponse =
    if (!storage.exists("architectures", request.architectureId))
      LoraManager.failure(
        s"architecture '${request.architectureId}' does not exist"
      )
    else
      request.source match {
        case LoraInstallSource.CivitaiFiles(_, _)
            if storage
              .get[Architecture]("architectures", request.architectureId)
              .exists(_.tool == RuntimeTool.LlamaCpp) =>
          LoraManager.failure(
            "Civitai hosts no chat model LoRAs: install a GGUF adapter from " +
              "HuggingFace or the disk"
          )
        case LoraInstallSource.CivitaiFiles(modelId, files) =>
          installFromCivitai(
            request.architectureId,
            modelId,
            files,
            request.grouping
          )
        case LoraInstallSource.HuggingFaceFiles(repo, filenames) =>
          installFromHuggingFace(
            request.architectureId,
            repo,
            filenames,
            request.grouping
          )
        case LoraInstallSource.ModelScopeFiles(repo, filenames) =>
          installFromModelScope(
            request.architectureId,
            repo,
            filenames,
            request.grouping
          )
        case LoraInstallSource.LocalFiles(paths) =>
          installFromDisk(request.architectureId, paths, request.grouping)
        case LoraInstallSource.Catalog(loraId) =>
          installFromCatalog(request.architectureId, loraId)
      }

  /** Why a file cannot be a LoRA of this architecture, if it cannot. A chat
    * model's LoRA must be GGUF: llama-server loads nothing else, so a
    * safetensors adapter would install and never apply
    * (`specs/35-assistant-loras.md`).
    */
  private def refusalOf(architectureId: String, name: String): Option[String] =
    if (!LoraManager.isWeightFile(name)) Some(s"'$name' is not a weight file")
    else if (
      storage
        .get[Architecture]("architectures", architectureId)
        .exists(_.tool == RuntimeTool.LlamaCpp) &&
      !LoraAdapters.isAdapterFile(name)
    )
      Some(
        s"'$name' is not a GGUF file: llama.cpp loads LoRA adapters as GGUF " +
          "only (convert a safetensors adapter with llama.cpp's " +
          "convert_lora_to_gguf.py first)"
      )
    else None

  /** The files ticked in one Civitai model, whatever version each belongs to
    * (`specs/33-lora-sources.md`). What they become — one LoRA, one each, or
    * files of one already installed — is the request's grouping.
    */
  private def installFromCivitai(
      architectureId: String,
      modelId: String,
      refs: List[CivitaiFileRef],
      grouping: LoraGrouping
  ): InstallLoraResponse =
    modelId.toIntOption.flatMap(civitaiClient.getModelDetail) match {
      case None =>
        LoraManager.failure(
          s"Civitai has no model '$modelId' (or the lookup failed)"
        )
      case Some(detail) =>
        val chosen = refs.flatMap { ref =>
          detail.modelVersions
            .find(_.id.toString == ref.versionId)
            .flatMap(version =>
              version.files
                .find(_.id.toString == ref.fileId)
                .filter(file => refusalOf(architectureId, file.name).isEmpty)
                .map(file => (version, file))
            )
        }
        if (chosen.isEmpty)
          LoraManager.failure(
            s"none of the chosen files of '${detail.name}' can be a LoRA of " +
              "this architecture"
          )
        else {
          val versions = chosen.map((version, _) => version).distinctBy(_.id)
          val fetched = chosen.map { (version, file) =>
            // Its own name tells it from another file of the same version;
            // that only matters when several of them are installed apart.
            val alone = chosen.count((v, _) => v.id == version.id) == 1
            LoraPlacement.Fetched(
              LoraFile(
                fileName = s"${file.id}-${file.name}",
                source = Civitai(
                  modelId = detail.id.toString,
                  versionId = version.id.toString,
                  fileId = file.id.toString,
                  filename = file.name
                ),
                stage = LoraManager.stageOf(file.name, version.name),
                sizeBytes = file.sizeKB.map(kb => (kb * 1024).toLong),
                sha256 = file.sha256
              ),
              ownLabel = civitaiLabel(
                detail,
                List(version),
                Option.unless(alone)(file)
              )
            )
          }
          placement.place(
            architectureId = architectureId,
            grouping = grouping,
            // The architecture is part of the identity: installing the same
            // Civitai model from two architectures' browsers yields two
            // independent entities, each holding only its own files.
            groupLabel = civitaiLabel(detail, versions, None),
            sourcePart = detail.id.toString,
            fetched = fetched,
            metadata = LoraPlacement.Metadata(
              fresh = lora =>
                lora.copy(
                  nsfw = detail.nsfw.getOrElse(false) ||
                    detail.nsfwLevel.exists(_ >= 8),
                  triggerWords = versions.flatMap(_.trainedWords).distinct,
                  tags = detail.tags,
                  description = detail.description.map(_.take(2000))
                ),
              refresh = lora =>
                lora.copy(
                  triggerWords = (lora.triggerWords ++ versions.flatMap(
                    _.trainedWords
                  )).distinct,
                  tags = detail.tags
                )
            ),
            onSaved = lora =>
              sidecars.writeSidecar(
                lorasRoot.resolve(lora.folderRelativePath),
                detail,
                versions.head,
                lora
              )
          )
        }
    }

  /** What a Civitai LoRA is called: the model's name, and the version's too
    * when the model publishes several and this LoRA holds only that one — which
    * is what lets two versions of it sit side by side in the picker and be
    * compared (François, 2026-09-18). `file` names the file as well, for when
    * one version's files are installed apart from each other.
    */
  private def civitaiLabel(
      detail: CivitaiModelDetail,
      versions: List[CivitaiModelVersion],
      file: Option[CivitaiModelFile]
  ): String = {
    val base = versions match {
      case List(one) if detail.modelVersions.sizeIs > 1 =>
        s"${detail.name} — ${one.name}"
      case _ => detail.name
    }
    file.map(one => s"$base — ${LoraManager.labelOf(one.name)}").getOrElse(base)
  }

  /** The files ticked in one HuggingFace repository. It carries no trigger
    * words, tags or nsfw flag: the label is the file's path without its
    * extension (the repository's own name when several travel together), the
    * rest is the user's to tune.
    */
  private def installFromHuggingFace(
      architectureId: String,
      repo: String,
      filenames: List[String],
      grouping: LoraGrouping
  ): InstallLoraResponse =
    fromRepository(
      architectureId,
      repo,
      filenames,
      grouping,
      filename =>
        huggingFace.lookup(HuggingFace(repo, filename)).map { info =>
          LoraFile(
            fileName = filename.split('/').last,
            source = HuggingFace(repo, filename),
            stage = LoraManager.stageOf(filename, repo),
            sizeBytes = info.lfs.flatMap(_.size).orElse(info.size),
            sha256 = info.sha256
          )
        }
    )

  /** The files ticked in one ModelScope repository (`specs/37-modelscope.md`),
    * labelled and identified like HuggingFace's: ModelScope lists their size
    * and sha256.
    */
  private def installFromModelScope(
      architectureId: String,
      repo: String,
      filenames: List[String],
      grouping: LoraGrouping
  ): InstallLoraResponse =
    fromRepository(
      architectureId,
      repo,
      filenames,
      grouping,
      filename =>
        modelScope.lookup(ModelScope(repo, filename)).map { info =>
          LoraFile(
            fileName = filename.split('/').last,
            source = ModelScope(repo, filename),
            stage = LoraManager.stageOf(filename, repo),
            sizeBytes = info.size,
            sha256 = info.sha256
          )
        }
    )

  /** What the two repository sites share: every ticked file looked up on the
    * site, then placed as the grouping says. One file that cannot be a LoRA
    * here refuses the whole install rather than half of it.
    */
  private def fromRepository(
      architectureId: String,
      repo: String,
      filenames: List[String],
      grouping: LoraGrouping,
      lookup: String => Either[String, LoraFile]
  ): InstallLoraResponse = {
    val refused = filenames.flatMap(name => refusalOf(architectureId, name))
    if (filenames.isEmpty) LoraManager.failure("no file was chosen")
    else if (refused.nonEmpty) LoraManager.failure(refused.head)
    else
      filenames.map(lookup).partitionMap(identity) match {
        case (reason :: _, _) => LoraManager.failure(reason)
        case (_, files)       =>
          placement.place(
            architectureId = architectureId,
            grouping = grouping,
            groupLabel =
              if (filenames.sizeIs == 1) LoraManager.labelOf(filenames.head)
              else repo.split('/').last,
            sourcePart = ModelCache.slug(repo.takeWhile(_ != '/')),
            fetched = filenames
              .zip(files)
              .map((name, file) =>
                LoraPlacement.Fetched(file, LoraManager.labelOf(name))
              )
          )
      }
  }

  /** Files on the drift host, copied rather than linked: the original may move
    * or vanish, and deleting the LoRA must not touch it.
    */
  private def installFromDisk(
      architectureId: String,
      rawPaths: List[String],
      grouping: LoraGrouping
  ): InstallLoraResponse = {
    val paths = rawPaths.map(raw => Paths.get(raw).toAbsolutePath.normalize)
    val names =
      paths.map(path => Option(path.getFileName).map(_.toString).getOrElse(""))
    val refusal = paths.zip(names).collectFirst {
      case (path, _) if !Files.isRegularFile(path) =>
        s"'$path' is not a file"
      case (path, _) if path.startsWith(lorasRoot.toAbsolutePath.normalize) =>
        s"'$path' is already in the LoRA store — adopt its folder from the " +
          "Model Cache instead"
      case (_, name) if refusalOf(architectureId, name).isDefined =>
        refusalOf(architectureId, name).get
    }
    if (paths.isEmpty) LoraManager.failure("no file was chosen")
    else
      refusal match {
        case Some(reason) => LoraManager.failure(reason)
        case None         =>
          placement.place(
            architectureId = architectureId,
            grouping = grouping,
            groupLabel =
              if (names.sizeIs == 1) LoraManager.labelOf(names.head)
              else LoraManager.labelOf(names.head),
            sourcePart = "local",
            fetched = paths.zip(names).map { (path, name) =>
              LoraPlacement.Fetched(
                LoraFile(
                  fileName = name,
                  source = Local(path.toString),
                  // The whole path: a file sorted into a `high_noise/` folder
                  // says so there and nowhere else.
                  stage = LoraManager.stageOf(path.toString),
                  sizeBytes = Some(Files.size(path))
                ),
                LoraManager.labelOf(name)
              )
            }
          )
      }
  }

  /** The catalog entry becomes the user's entity as it is — label, description,
    * strength and stages chosen by whoever wrote the entry.
    */
  private def installFromCatalog(
      architectureId: String,
      loraId: String
  ): InstallLoraResponse =
    catalog.find(_.id == loraId) match {
      case None => LoraManager.failure(s"drift offers no LoRA '$loraId'")
      case Some(entry) if entry.architectureId != architectureId =>
        LoraManager.failure(
          s"'${entry.label}' is a LoRA for ${entry.architectureId}, not $architectureId"
        )
      case Some(entry) =>
        val lora = storage
          .get[Lora]("loras", entry.id)
          .getOrElse(entry.copy(createdAt = System.currentTimeMillis()))
        storage.save("loras", lora.id, lora)
        lora.files.foreach(downloads.queueDownload(lora, _))
        InstallLoraResponse(lora = Some(lora))
    }
}
