package drift.frontend.services

import drift.shared.*

import com.raquo.laminar.api.L.*

object PromptTemplateService {
  type Command = EntityService.Command[PromptTemplate]
  val Command: EntityService.Command.type = EntityService.Command
  type Event = EntityService.Event[PromptTemplate]
  val Event: EntityService.Event.type = EntityService.Event
}

/** The prompt library (`specs/32-prompt-library.md`): every template, loaded
  * once for the app since the assistant, redraw and Settings all read it.
  */
class PromptTemplateService
    extends EntityService[PromptTemplate](
      ApiClient.stream(listPromptTemplates),
      ApiClient.stream(createPromptTemplate),
      ApiClient.stream(updatePromptTemplate),
      ApiClient.stream(deletePromptTemplate),
      _.id,
      EntityWording(
        loading = "Loading the prompt library",
        creating = "Saving the prompt",
        saving = "Saving the prompt",
        deleting = "Deleting the prompt",
        missing = _ => "the server has no such prompt.",
        refused = id =>
          s"the server refused to delete '$id' — built-ins cannot be deleted."
      )
    ) {
  val templates: Signal[List[PromptTemplate]] = entities.signal

  /** Built-ins first, then the user's, by label. */
  private def ordered(all: List[PromptTemplate]): List[PromptTemplate] =
    all.sortBy(t => (!t.builtIn, t.label.toLowerCase))

  /** The list with a template in its place, whether it was there or not. */
  private def placed(
      all: List[PromptTemplate],
      template: PromptTemplate
  ): List[PromptTemplate] =
    ordered(all.filterNot(_.id == template.id) :+ template)

  override protected def arranged(
      loaded: List[PromptTemplate]
  ): List[PromptTemplate] = ordered(loaded)

  override protected def withCreated(
      all: List[PromptTemplate],
      created: PromptTemplate
  ): List[PromptTemplate] = placed(all, created)

  override protected def withUpdated(
      all: List[PromptTemplate],
      id: String,
      saved: PromptTemplate
  ): List[PromptTemplate] = placed(all, saved)

  def ofKind(kind: PromptKind): Signal[List[PromptTemplate]] =
    templates.map(_.filter(_.kind == kind)).distinct

  def now(id: String): Option[PromptTemplate] =
    entities.now().find(_.id == id)

  /** The template of an id, else the kind's built-in, else nothing — the caller
    * then reports it rather than sending an empty prompt.
    */
  def resolve(kind: PromptKind, id: Option[String]): Option[PromptTemplate] =
    id.flatMap(now)
      .filter(_.kind == kind)
      .orElse(entities.now().find(t => t.builtIn && t.kind == kind))
}
