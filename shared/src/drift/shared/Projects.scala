package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.model.StatusCode
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Projects and prompt versions (`specs/19-projects-and-prompt-versions.md`). A
  * project is one thing being made; a version is a recipe that actually ran —
  * every generation inside a project belongs to one, and a generation whose
  * recipe differs from the selected version creates the next.
  */

/** Who wrote the recipe: the user in the form, or the assistant's proposal
  * applied to it.
  */
enum VersionOrigin derives CanEqual {
  case Manual, Assistant
}
object VersionOrigin {
  given Schema[VersionOrigin] =
    Schema.derivedEnumeration[VersionOrigin].defaultStringBased
}

/** What a project makes (`specs/31-project-kinds.md`): its workspace offers
  * only the configurations whose architecture makes it. A text project is a
  * kept conversation with a chat model and nothing else
  * (`specs/41-text-projects.md`).
  */
enum ProjectKind derives CanEqual {
  case Image, Video, Text

  /** "image", "video", "text" — for sentences. */
  def noun: String = toString.toLowerCase

  /** Whether an architecture's models make this kind, read from its tags: an
    * edit model makes images too, a chat model text; an upscaler makes none.
    */
  def accepts(architecture: Architecture): Boolean = this match {
    case Image =>
      architecture.tags.exists(tag =>
        tag == ArchitectureTags.Image || tag == ArchitectureTags.Edit
      )
    case Video => architecture.tags.contains(ArchitectureTags.Video)
    case Text  => architecture.tags.contains(ArchitectureTags.Llm)
  }
}
object ProjectKind {
  given Schema[ProjectKind] =
    Schema.derivedEnumeration[ProjectKind].defaultStringBased
}

/** One recipe: the full request as submitted — prompt, negative prompt, size,
  * sampling, LoRAs, hires or frames, tiling — and the configuration it ran on.
  * Shaped like the generation that made it, job kind and that kind's
  * parameters, so a version reads like the sidecar of its generations and seeds
  * the form the way reusing one of them does. The seed and the input images are
  * not part of a recipe's identity: a re-roll is the same version, and the
  * inputs are recorded as URLs beside the outputs.
  */
case class PromptVersion(
    id: String,
    number: Int,
    /** What changed from the parent, written by drift ("changed prompt,
      * sampling"), editable afterwards.
      */
    note: String,
    parentId: Option[String],
    origin: VersionOrigin,
    createdAt: Long,
    runConfigurationId: String,
    /** "img_gen" or "vid_gen": which of the two parameters below is set. */
    kind: String,
    imageParameters: Option[ImageGenerationParameters] = None,
    videoParameters: Option[VideoGenerationParameters] = None
) {
  def prompt: String = imageParameters
    .map(_.prompt)
    .orElse(videoParameters.map(_.prompt))
    .getOrElse("")

  def negativePrompt: String = imageParameters
    .map(_.negativePrompt)
    .orElse(videoParameters.map(_.negativePrompt))
    .getOrElse("")
}
object PromptVersion {
  given Schema[PromptVersion] = Schema.derived
}

case class Project(
    id: String,
    label: String,
    /** What is being made, in the user's words — the assistant reads it. */
    brief: String,
    createdAt: Long,
    lastUsedAt: Long,
    versions: List[PromptVersion] = List.empty,
    /** Hidden from the projects list unless asked for, and its workspace offers
      * the NSFW LoRAs — a project is SFW unless flagged.
      */
    nsfw: Boolean = false,
    /** Chosen at creation, changeable in the workspace. A project saved before
      * kinds existed has no such field and reads as an image project — every
      * one of them was.
      */
    kind: ProjectKind = ProjectKind.Image,
    /** The output the user chose to show on the project's card; none means the
      * newest result, as it was before covers could be chosen.
      */
    cover: Option[ProjectCover] = None,
    /** The assistant system template chosen for this project's conversation,
      * and the compaction template; none means the built-in
      * (`specs/32-prompt-library.md`).
      */
    assistantTemplateId: Option[String] = None,
    compactionTemplateId: Option[String] = None
)
object Project {
  given Schema[Project] = Schema.derived
  // Snake case like the sidecars, so an embedded request reads the same in
  // both files; no discriminator, the enum has only singleton cases.
  given JsonValueCodec[Project] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
  given JsonValueCodec[List[Project]] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
  given JsonValueCodec[Option[Project]] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
}

/** A chosen cover (`specs/19-projects-and-prompt-versions.md`): one output,
  * named the way the outputs root files it, with its type so the cover endpoint
  * can tell whether it still suits the project's kind.
  */
case class ProjectCover(date: String, fileName: String, mimeType: String) {
  def isOf(output: GenerationOutput): Boolean =
    output.date == date && output.fileName == fileName
}
object ProjectCover {
  given Schema[ProjectCover] = Schema.derived

  def of(output: GenerationOutput): ProjectCover =
    ProjectCover(output.date, output.fileName, output.mimeType)
}

/** What a submission says about its project, as query parameters beside the
  * native request: which project, which version was selected in the form, and
  * whether the prompt came from the assistant. The backend decides whether the
  * submission is that version or the next one.
  */
case class SubmitContext(
    projectId: Option[String] = None,
    versionId: Option[String] = None,
    origin: Option[String] = None
)

private val projectsBase = endpoint.in("api")

val listProjects: PublicEndpoint[Unit, Unit, List[Project], Any] =
  projectsBase.get.in("projects").out(jsonBody[List[Project]])

val getProject: PublicEndpoint[String, Unit, Option[Project], Any] =
  projectsBase.get.in("projects" / path[String]).out(jsonBody[Option[Project]])

val createProject: PublicEndpoint[Project, Unit, Project, Any] =
  projectsBase.post.in("projects").in(jsonBody[Project]).out(jsonBody[Project])

val updateProject
    : PublicEndpoint[(String, Project), Unit, Option[Project], Any] =
  projectsBase.put
    .in("projects" / path[String])
    .in(jsonBody[Project])
    .out(jsonBody[Option[Project]])

/** Deletes the project and its versions; the generations it made keep their
  * files and sidecars, only the listing by project is gone. Deleting those too
  * is a separate call — [[deleteProjectGenerations]] — so that "the project,
  * not the images" stays possible.
  */
val deleteProject: PublicEndpoint[String, Unit, Boolean, Any] =
  projectsBase.delete.in("projects" / path[String]).out(jsonBody[Boolean])

/** Deletes every generation tagged with the project — output files,
  * externalized inputs and sidecars — and answers with the ids that went. The
  * project document itself is untouched: the frontend asks for both when the
  * user confirms both.
  */
val deleteProjectGenerations: PublicEndpoint[String, Unit, List[String], Any] =
  projectsBase.delete
    .in("projects" / path[String] / "generations")
    .out(jsonBody[List[String]])

/** A project's cover - the output chosen for it, else the newest result it made -
  * as bytes an `img` or a `video` can point straight at (François, 2026-09-10).
  * An image project's is its newest image, as a downscaled JPEG rather than the
  * original: a cover sits in a 20rem tile, and the generations behind it are
  * multi-megabyte PNGs. A video project's is its newest video's first frame,
  * the same JPEG, so the list loads no video (`specs/31-project-kinds.md`, bug
  * 37).
  *
  * 404 when the project has made nothing yet, which is not an error - the card
  * shows its placeholder and says so.
  */
val getProjectCover: PublicEndpoint[String, Unit, (Array[Byte], String), Any] =
  projectsBase.get
    .in("projects" / path[String] / "cover")
    .errorOut(statusCode(StatusCode.NotFound))
    .out(byteArrayBody)
    .out(header[String]("Content-Type"))

/** Where the list's `img` points; the browser does the fetching and the
  * caching, so nothing about covers reaches the frontend's state.
  */
def projectCoverPath(projectId: String): String =
  s"/api/projects/$projectId/cover"

/** Every recorded generation tagged with the project, newest first — a walk of
  * the sidecars, since they are the durable record.
  */
val listProjectGenerations
    : PublicEndpoint[String, Unit, List[Generation], Any] =
  projectsBase.get
    .in("projects" / path[String] / "generations")
    .out(jsonBody[List[Generation]])

/** A prompt pair: what the assistant proposes, and what a proposal is diffed
  * against (`specs/20`). Shared because a conversation stores, for each reply,
  * the working prompt it was asked about.
  */
case class PromptProposal(
    prompt: String,
    negativePrompt: String,
    /** `W:H`, when the reply carried one (an Ideogram caption's `aspect_ratio`,
      * taken out of the prompt): applying the proposal sets the form's size to
      * it (`specs/32-prompt-library.md`).
      */
    aspectRatio: Option[String] = None
)
object PromptProposal {
  given Schema[PromptProposal] = Schema.derived
}

/** One message of a project's conversation with the assistant (`specs/20`).
  * `role` is `user`, `assistant` or `summary` — a summary stands for the
  * messages a compaction moved to the archive. Images are not stored: a user
  * message keeps the preview URLs it showed, and its text already carries each
  * attachment's recorded parameters.
  */
case class ConversationMessage(
    id: Int,
    role: String,
    text: String,
    reasoning: String = "",
    previews: List[String] = List.empty,
    /** The working prompt this reply was asked about — what its proposal is
      * diffed against.
      */
    promptBase: Option[PromptProposal] = None,
    promptTokens: Option[Int] = None,
    completionTokens: Option[Int] = None,
    error: Option[String] = None,
    createdAt: Long = 0,
    /** The proposal read from an assistant reply when it arrived, kept so a
      * reload shows the same card whatever template is chosen now (`specs/32`).
      */
    proposal: Option[PromptProposal] = None
)
object ConversationMessage {
  given Schema[ConversationMessage] = Schema.derived
}

/** A project's conversation (`specs/20`): the live transcript, what compaction
  * and restarts moved out of it, and when it was last compacted. Stored under
  * `conversations/<projectId>.json` rather than beside the project, since every
  * file in `projects/` is read as a `Project`.
  */
case class Conversation(
    projectId: String,
    messages: List[ConversationMessage] = List.empty,
    archive: List[ConversationMessage] = List.empty,
    compactedAt: Option[Long] = None
)
object Conversation {
  given Schema[Conversation] = Schema.derived
  // Snake case, like the project document it belongs to.
  given JsonValueCodec[Conversation] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
}

/** The project's conversation — an empty one when nothing is stored yet. */
val getConversation: PublicEndpoint[String, Unit, Conversation, Any] =
  projectsBase.get
    .in("projects" / path[String] / "conversation")
    .out(jsonBody[Conversation])

/** Replaces the project's conversation: the frontend owns the transcript and
  * saves it whole after every change.
  */
val saveConversation
    : PublicEndpoint[(String, Conversation), Unit, Conversation, Any] =
  projectsBase.put
    .in("projects" / path[String] / "conversation")
    .in(jsonBody[Conversation])
    .out(jsonBody[Conversation])
