package drift.frontend.pages.gallery

import drift.frontend.Page
import drift.frontend.components.*
import drift.frontend.pages.gallery.VisionAssistants.State
import drift.frontend.services.LaunchPrerequisites

import com.raquo.laminar.api.L.*

/** What the auto redraw card shows while no assistant can read the picture
  * (`bugs/43`): which one to start, from where the user is — or, when a chat
  * model that does not read images is live, the one action that stops it and
  * starts one that does in its place. Never a second model started beside it,
  * and never the first stopped without that click: a conversation may be going
  * on with it.
  */
class VisionAssistantStarter(
    assistants: VisionAssistants,
    prerequisites: LaunchPrerequisites,
    /** Whether the picture is read once the assistant serves. */
    waiting: Signal[Boolean],
    /** A start was asked for: the card reads the picture when it is ready. */
    onStarted: () => Unit
) extends Component {

  private val pickedVar = Var("")

  /** The one to start: the one picked, else the last one run. */
  private val chosen: Signal[String] =
    assistants.configurations
      .combineWith(pickedVar.signal)
      .map((list, picked) =>
        list
          .find(_.id == picked)
          .orElse(list.headOption)
          .fold("")(_.id)
      )
      .distinct

  private def configurationSelect: HtmlElement =
    div(
      cls := "select is-small",
      cls("is-hidden") <-- assistants.configurations.map(_.sizeIs < 2),
      select(
        onChange.mapToValue --> pickedVar,
        children <-- assistants.configurations.map(
          _.map(configuration =>
            option(
              value := configuration.id,
              selected <-- chosen.map(_ == configuration.id),
              configuration.label
            )
          )
        )
      )
    )

  /** The select and the button that starts the chosen one — the download of
    * what it lacks until it can start (`specs/46`).
    */
  private def start(
      replacing: Option[VisionAssistants.Live]
  ): HtmlElement =
    span(
      cls := "is-inline-flex is-align-items-center",
      styleAttr := "gap: 0.5rem;",
      configurationSelect,
      LaunchOrDownload(
        prerequisites.of(chosen),
        prerequisites,
        button(
          cls := (if (replacing.isDefined) "button is-small is-warning"
                  else "button is-small is-link"),
          child.text <-- assistants.configurations.combineWith(chosen).map {
            (list, id) =>
              val label = list.find(_.id == id).fold("it")(_.label)
              replacing.fold(s"▶ Start $label and read the picture")(live =>
                s"⏹ Stop ${live.label}, start $label and read the picture"
              )
          },
          onClick.compose(_.sample(chosen)) --> (id =>
            if (id.nonEmpty) {
              assistants.start(id, replacing)
              onStarted()
            }
          )
        )
      ).element
    )

  private def line(text: String, more: Modifier[HtmlElement]*): HtmlElement =
    div(
      cls := "is-size-7 auto-redraw-assistant",
      span(cls := "text-secondary mr-2", text),
      more
    )

  lazy val element: HtmlElement = div(
    child <-- assistants.state.map {
      case State.Ready(_)         => emptyNode
      case State.Loading(session) =>
        line(
          "",
          child.text <-- waiting.map(read =>
            s"⏳ ${session.label} is loading" +
              (if (read) " — the picture is read as soon as it is ready."
               else "…")
          )
        )
      case State.Other(session) =>
        line(
          s"${session.label} is running and does not read images.",
          start(Some(session))
        )
      case State.NoneRunning =>
        line("No assistant is running.", start(None))
      case State.NoneInstalled =>
        line(
          "No chat configuration reads images: create one whose model has a " +
            "vision projector (mmproj).",
          button(
            cls := "button is-small",
            "Models →",
            onClick --> (_ => Page.Models.navigate())
          )
        )
    },
    MemoryWarningView(
      assistants.state.map {
        case State.Ready(session)   => session.memoryWarning
        case State.Loading(session) => session.memoryWarning
        case _                      => None
      }
    ).element
  )
}
