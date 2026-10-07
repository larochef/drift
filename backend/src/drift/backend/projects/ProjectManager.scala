package drift.backend.projects

import drift.backend.storage.StorageService
import drift.shared.*

/** The versioning rule of `specs/19-projects-and-prompt-versions.md`: every
  * generation inside a project belongs to a version, image or video alike
  * (`specs/31-project-kinds.md`). A submission whose recipe equals the selected
  * version's — on the same configuration, seed and input images aside — is that
  * version; anything else appends the next, with the selected one as parent and
  * a note naming what changed.
  */
final class ProjectManager(storage: StorageService) {

  /** The project and version a generation belongs to, `None` outside a project.
    * Compares and stores the *recorded* request (inputs as URLs), so a version
    * reads exactly like the sidecar of its generations.
    */
  def versionFor(
      context: SubmitContext,
      generation: Generation
  ): Either[String, Option[(String, String)]] =
    context.projectId match {
      case None            => Right(None)
      case Some(projectId) =>
        storage.get[Project]("projects", projectId) match {
          case None          => Left(s"project '$projectId' does not exist")
          case Some(project) =>
            val selected =
              context.versionId.flatMap(id => project.versions.find(_.id == id))
            selected match {
              case Some(version)
                  if version.runConfigurationId == generation.runConfigurationId &&
                    ProjectManager.sameRecipe(version, generation) =>
                touch(project)
                Right(Some((projectId, version.id)))
              case _ =>
                val number = project.versions
                  .map(_.number)
                  .maxOption
                  .getOrElse(0) + 1
                val version = PromptVersion(
                  id = s"$projectId-v$number",
                  number = number,
                  note = ProjectManager.describeChange(selected, generation),
                  parentId = selected.map(_.id),
                  origin =
                    if (context.origin.contains("assistant"))
                      VersionOrigin.Assistant
                    else VersionOrigin.Manual,
                  createdAt = System.currentTimeMillis(),
                  runConfigurationId = generation.runConfigurationId,
                  kind = generation.kind,
                  imageParameters = generation.imageParameters,
                  videoParameters = generation.videoParameters
                )
                storage.save(
                  "projects",
                  projectId,
                  project.copy(
                    versions = project.versions :+ version,
                    lastUsedAt = version.createdAt
                  )
                )
                Right(Some((projectId, version.id)))
            }
        }
    }

  /** The version of `projectId` a generation made elsewhere joins when it is
    * moved there: the one already holding its recipe on its configuration, else
    * a new one — twenty pictures of one recipe moved together are one version,
    * not twenty.
    */
  def adopt(
      projectId: String,
      generation: Generation
  ): Either[String, Option[(String, String)]] =
    versionFor(
      SubmitContext(
        projectId = Some(projectId),
        versionId = storage
          .get[Project]("projects", projectId)
          .flatMap(
            _.versions.find(version =>
              version.runConfigurationId == generation.runConfigurationId &&
                ProjectManager.sameRecipe(version, generation)
            )
          )
          .map(_.id),
        origin = Some("manual")
      ),
      generation
    )

  private def touch(project: Project): Unit =
    storage.save(
      "projects",
      project.id,
      project.copy(lastUsedAt = System.currentTimeMillis())
    )
}

object ProjectManager {

  /** The recipe without what varies between runs of the same recipe: the seed
    * and the inputs. A batch count is one of those too (`specs/14`): four
    * images in one run are four re-rolls of the recipe, the same thing the
    * blanked seed already says, so asking for more of them must not append a
    * "changed batch" version. An image recipe never equals a video one.
    */
  private def recipe(
      imageParameters: Option[ImageGenerationParameters],
      videoParameters: Option[VideoGenerationParameters]
  ): Option[Product] =
    imageParameters
      .map(
        _.copy(
          seed = -1,
          batchCount = 1,
          initImage = None,
          maskImage = None,
          refImages = List.empty
        )
      )
      .orElse(
        videoParameters.map(p =>
          p.copy(
            seed = -1,
            initImage = None,
            endImage = None,
            controlFrames = List.empty,
            // A guide's frame is part of the recipe; its medium, like every
            // input, is a file that differs from run to run.
            references = List.empty,
            guides = p.guides.map(_.copy(media = "")),
            controlVideo = None,
            controlMask = None,
            sourceVideo = None
          )
        )
      )

  private def recipeOf(version: PromptVersion): Option[Product] =
    recipe(version.imageParameters, version.videoParameters)

  private def recipeOf(generation: Generation): Option[Product] =
    recipe(generation.imageParameters, generation.videoParameters)

  def sameRecipe(version: PromptVersion, generation: Generation): Boolean =
    recipeOf(version) == recipeOf(generation)

  /** The note of a new version: which fields differ from the parent's recipe,
    * in words, plus the configuration when that changed too.
    */
  def describeChange(
      parent: Option[PromptVersion],
      generation: Generation
  ): String = parent match {
    case None           => "first version"
    case Some(previous) =>
      val fields = (recipeOf(previous), recipeOf(generation)) match {
        case (Some(before), Some(after))
            if before.productPrefix == after.productPrefix =>
          before.productElementNames.toList
            .zip(
              before.productIterator.toList.zip(after.productIterator.toList)
            )
            .collect { case (name, (x, y)) if x != y => name }
            .map(fieldWord)
            .distinct
        // An image recipe after a video one, or the other way round: the
        // fields do not line up, and the kind is the change worth naming.
        case _ => List("kind")
      }
      val configuration =
        Option.when(
          previous.runConfigurationId != generation.runConfigurationId
        )(
          s"on ${generation.runConfigurationId}"
        )
      (Option.when(fields.nonEmpty)(s"changed ${fields.mkString(", ")}") ++
        configuration).mkString("; ") match {
        case ""   => "same recipe"
        case note => note
      }
  }

  private def fieldWord(name: String): String = name match {
    case "prompt"                             => "prompt"
    case "negativePrompt"                     => "negative prompt"
    case "width" | "height"                   => "size"
    case "sampleParams"                       => "sampling"
    case "highNoiseSampleParams"              => "high-noise sampling"
    case "lora"                               => "LoRAs"
    case "hires"                              => "hires"
    case "vaeTilingParams"                    => "VAE tiling"
    case "strength"                           => "strength"
    case "batchCount"                         => "batch"
    case "clipSkip"                           => "clip skip"
    case "videoFrames"                        => "frames"
    case "fps"                                => "frame rate"
    case "moeBoundary"                        => "MoE boundary"
    case "vaceStrength"                       => "VACE strength"
    case "outputFormat" | "outputCompression" => "output"
    case other                                => other
  }
}
