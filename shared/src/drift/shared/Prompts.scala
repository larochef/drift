package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Where a prompt template is used (`specs/32-prompt-library.md`); `Edit` is
  * what an edit's tiles are told before the instruction
  * (`specs/39-seamless-edit.md`), `RedrawDiagnosis` what the assistant is asked
  * when it reads a picture for its redraw (`specs/52-auto-redraw.md`).
  */
enum PromptKind derives CanEqual {
  case AssistantSystem, RedrawRestoration, Compaction, Edit, RedrawDiagnosis
}
object PromptKind {
  given Schema[PromptKind] =
    Schema.derivedEnumeration[PromptKind].defaultStringBased
}

/** How a proposal is read out of an assistant reply: the fenced `prompt` and
  * `negative` blocks, or the last JSON object in the reply, minified into the
  * prompt (an Ideogram caption).
  */
enum ProposalFormat derives CanEqual {
  case Fenced, Json
}
object ProposalFormat {
  given Schema[ProposalFormat] =
    Schema.derivedEnumeration[ProposalFormat].defaultStringBased
}

/** Which of drift's additions follow an assistant system template. Off for a
  * template whose job the addition would contradict — the Ideogram caption
  * writer converts an idea, so the editing rules stay out.
  */
case class AssistantAppends(
    brief: Boolean,
    promptingNotes: Boolean,
    workingPrompt: Boolean,
    rules: Boolean,
    cfgNote: Boolean,
    /** The form's aspect ratio, for templates that want one (Ideogram's expects
      * it with the idea).
      */
    aspectRatio: Boolean
)
object AssistantAppends {

  /** What a template appends unless it says otherwise: everything but the
    * aspect ratio.
    */
  val standard: AssistantAppends = AssistantAppends(
    brief = true,
    promptingNotes = true,
    workingPrompt = true,
    rules = true,
    cfgNote = true,
    aspectRatio = false
  )

  given Schema[AssistantAppends] = Schema.derived
}

/** One variant of a prompt drift puts in front of a model. Built-ins come from
  * the reference data and are re-seeded on every start; a user variant is a
  * copy.
  */
case class PromptTemplate(
    id: String,
    kind: PromptKind,
    label: String,
    text: String,
    appends: AssistantAppends,
    proposalFormat: ProposalFormat,
    builtIn: Boolean
)
object PromptTemplate {

  /** Ids of the built-in templates every install has. */
  val DefaultAssistantId = "drift-assistant"
  val DefaultRedrawId = "redraw-restoration"
  val DefaultCompactionId = "drift-compaction"
  val DefaultEditId = "edit-seamless"
  val DefaultRedrawDiagnosisId = "redraw-diagnosis"

  given JsonValueCodec[PromptTemplate] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[PromptTemplate]] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[PromptTemplate]] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
  given Schema[PromptTemplate] = Schema.derived
}

val listPromptTemplates: PublicEndpoint[Unit, Unit, List[PromptTemplate], Any] =
  base.get
    .in("prompt-templates")
    .out(jsonBody[List[PromptTemplate]])

val getPromptTemplate
    : PublicEndpoint[String, Unit, Option[PromptTemplate], Any] =
  base.get
    .in("prompt-templates" / path[String])
    .out(jsonBody[Option[PromptTemplate]])

val createPromptTemplate
    : PublicEndpoint[PromptTemplate, Unit, PromptTemplate, Any] =
  base.post
    .in("prompt-templates")
    .in(jsonBody[PromptTemplate])
    .out(jsonBody[PromptTemplate])

val updatePromptTemplate: PublicEndpoint[
  (String, PromptTemplate),
  Unit,
  Option[PromptTemplate],
  Any
] =
  base.put
    .in("prompt-templates" / path[String])
    .in(jsonBody[PromptTemplate])
    .out(jsonBody[Option[PromptTemplate]])

val deletePromptTemplate: PublicEndpoint[String, Unit, Boolean, Any] =
  base.delete
    .in("prompt-templates" / path[String])
    .out(jsonBody[Boolean])
