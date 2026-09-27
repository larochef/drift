package drift.frontend.pages.settings

import drift.frontend.components.{Component, ScrollLock}

import com.raquo.laminar.api.L.*

/** Adding a runtime is a modal with the ways of doing it behind tabs (François,
  * 2026-09-10): the page itself lists what is installed, which is what it is
  * usually consulted for, and the forms are a step away rather than two
  * sections always open below. The forms are built once by the section, so
  * switching tabs - or closing and reopening - does not lose what was typed.
  */
class AddRuntimeModal(
    installForm: => InstallRuntimeForm,
    runnerForm: => InstallRunnerForm,
    adoptForm: => AdoptRuntimeForm
) extends Component {
  private val open = Var(false)
  private val tab = Var("install")

  /** Opens the modal on the install tab. */
  def show(): Unit = {
    tab.set("install")
    open.set(true)
  }

  def hide(): Unit = open.set(false)

  private def tabItem(key: String, labelText: String): HtmlElement = li(
    cls <-- tab.signal.map(current => if (current == key) "is-active" else ""),
    a(onClick --> (_ => tab.set(key)), labelText)
  )

  lazy val element: HtmlElement = div(
    child <-- open.signal.map {
      case false => emptyNode
      case true  =>
        div(
          cls := "modal is-active add-runtime-modal",
          ScrollLock.whileMounted,
          documentEvents(_.onKeyDown).filter(_.key == "Escape")
            --> (_ => hide()),
          div(
            cls := "modal-background",
            onClick --> (_ => hide())
          ),
          div(
            cls := "modal-card",
            headerTag(
              cls := "modal-card-head",
              p(cls := "modal-card-title", "Add a runtime"),
              button(
                cls := "delete",
                aria.label := "close",
                onClick --> (_ => hide())
              )
            ),
            sectionTag(
              cls := "modal-card-body",
              div(
                cls := "tabs",
                ul(
                  tabItem("install", "Install a runtime"),
                  tabItem("runner", "Install the drift runner"),
                  tabItem("adopt", "Adopt an existing install")
                )
              ),
              child <-- tab.signal.map {
                case "adopt"  => adoptForm.element
                case "runner" => runnerForm.element
                case _        => installForm.element
              }
            ),
            footerTag(
              cls := "modal-card-foot",
              button(
                cls := "button",
                "Close",
                onClick --> (_ => hide())
              )
            )
          )
        )
    }
  )
}
