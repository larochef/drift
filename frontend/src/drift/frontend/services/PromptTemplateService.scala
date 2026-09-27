package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object PromptTemplateService {
  enum Command {
    case Load
    case Create(template: PromptTemplate)
    case Update(template: PromptTemplate)
    case Delete(id: String)
  }
}

/** The prompt library (`specs/32-prompt-library.md`): every template, loaded
  * once for the app since the assistant, redraw and Settings all read it.
  */
class PromptTemplateService extends ServiceErrors {
  import PromptTemplateService.Command

  private val listFn = ApiClient.stream(drift.shared.listPromptTemplates)
  private val createFn = ApiClient.stream(drift.shared.createPromptTemplate)
  private val updateFn = ApiClient.stream(drift.shared.updatePromptTemplate)
  private val deleteFn = ApiClient.stream(drift.shared.deletePromptTemplate)

  private val _templates = Var(List.empty[PromptTemplate])
  val templates: Signal[List[PromptTemplate]] = _templates.signal

  private val cmdBus = new EventBus[Command]
  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  /** Built-ins first, then the user's, by label. */
  private def ordered(all: List[PromptTemplate]): List[PromptTemplate] =
    all.sortBy(t => (!t.builtIn, t.label.toLowerCase))

  def ofKind(kind: PromptKind): Signal[List[PromptTemplate]] =
    templates.map(_.filter(_.kind == kind)).distinct

  def now(id: String): Option[PromptTemplate] =
    _templates.now().find(_.id == id)

  /** The template of an id, else the kind's built-in, else nothing — the caller
    * then reports it rather than sending an empty prompt.
    */
  def resolve(kind: PromptKind, id: Option[String]): Option[PromptTemplate] =
    id.flatMap(now)
      .filter(_.kind == kind)
      .orElse(_templates.now().find(t => t.builtIn && t.kind == kind))

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => listFn(()).recoverToTry)
      --> Observer[Try[List[PromptTemplate]]] {
        case Success(all) =>
          clearError()
          _templates.set(ordered(all))
        case Failure(err) => reportFailure("Loading the prompt library", err)
      },
    cmdBus.events
      .collect { case Command.Create(t) => t }
      .flatMapMerge(t => createFn(t).recoverToTry)
      --> Observer[Try[PromptTemplate]] {
        case Success(t) =>
          clearError()
          _templates.update(all => ordered(all.filterNot(_.id == t.id) :+ t))
        case Failure(err) => reportFailure("Saving the prompt", err)
      },
    cmdBus.events
      .collect { case Command.Update(t) => t }
      .flatMapMerge(t => updateFn((t.id, t)).recoverToTry)
      --> Observer[Try[Option[PromptTemplate]]] {
        case Success(Some(t)) =>
          clearError()
          _templates.update(all => ordered(all.filterNot(_.id == t.id) :+ t))
        case Success(None) =>
          reportFailure("Saving the prompt", "the server has no such prompt.")
        case Failure(err) => reportFailure("Saving the prompt", err)
      },
    cmdBus.events
      .collect { case Command.Delete(id) => id }
      .flatMapMerge(id => deleteFn(id).map((id, _)).recoverToTry)
      --> Observer[Try[(String, Boolean)]] {
        case Success((id, true)) =>
          clearError()
          _templates.update(_.filterNot(_.id == id))
        case Success((id, false)) =>
          reportFailure(
            "Deleting the prompt",
            s"the server refused to delete '$id' — built-ins cannot be deleted."
          )
        case Failure(err) => reportFailure("Deleting the prompt", err)
      }
  )
}
