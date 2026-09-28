package drift.frontend.pages.inference

import drift.frontend.components.*
import drift.frontend.services.{
  BrowserServices,
  LoraService,
  RunConfigurationService
}
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Creating a run configuration: `RunConfigurationForm` in a modal, from the
  * run configurations page and from a project's model pickers (François,
  * 2026-09-28: a project's model is made where it is picked). The form outlives
  * the modal, so closing and reopening keeps what was filled in; a created
  * configuration clears it.
  */
class NewRunConfigurationModal(
    service: RunConfigurationService,
    /** The architectures offered: the tools the caller lists. */
    architectures: Signal[List[Architecture]],
    loraService: LoraService,
    browsers: BrowserServices,
    assistantTemplates: Signal[List[PromptTemplate]],
    runtimes: Signal[List[Runtime]],
    onClose: () => Unit,
    /** Once the server has stored it, before the modal closes. */
    onCreated: RunConfiguration => Unit = _ => (),
    heading: String = "New run configuration"
) extends Component {

  private val form = RunConfigurationForm(
    architectures,
    service.modelService,
    loraService,
    browsers,
    assistantTemplates,
    runtimes
  )

  /** The id sent, so only its own creation closes the modal. */
  private var awaiting = Option.empty[String]

  private def create(): Unit =
    if (form.valid) {
      val configuration = form.snapshot()
      awaiting = Some(configuration.id)
      service.push(RunConfigurationService.Command.Create(configuration))
    }

  lazy val element: HtmlElement = NewRunConfigurationModal.frame(
    service,
    heading,
    form.element,
    button(
      cls := "button is-success",
      span(cls := "plus-icon", "+"),
      " Create",
      onClick --> (_ => create())
    ),
    onClose,
    service.events --> Observer[RunConfigurationService.Event] {
      case RunConfigurationService.Event.Created(created)
          if awaiting.contains(created.id) =>
        awaiting = None
        form.reset()
        onCreated(created)
        onClose()
      case _ => ()
    }
  )
}

object NewRunConfigurationModal {

  /** Creating or editing a configuration: the form in a modal, with the
    * service's errors inside it — a refused save would otherwise explain itself
    * behind the backdrop — and the confirming button beside Cancel.
    */
  def frame(
      service: RunConfigurationService,
      title: String,
      form: HtmlElement,
      confirm: HtmlElement,
      onCancel: () => Unit,
      extra: Mod[HtmlElement] = emptyNode
  ): HtmlElement =
    BrowserModal(
      title = Val(title),
      body = Seq(ErrorBanner(service), form),
      onCancel = onCancel,
      footerRight = div(
        cls := "buttons",
        confirm,
        button(cls := "button", "Cancel", onClick --> (_ => onCancel()))
      ),
      modalMods = Seq(
        documentEvents(_.onKeyDown).filter(_.key == "Escape")
          --> (_ => onCancel()),
        extra
      ),
      cardMods = Seq(styleAttr := "width: min(60rem, 95vw);")
    ).element
}
