package drift.frontend.components

import drift.frontend.services.{BrowserServices, LoraService}
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Installs LoRAs for one architecture (`specs/09-lora-management.md`,
  * `specs/33-lora-sources.md`): one button, opening one browser where the
  * source — Civitai, HuggingFace, ModelScope or this machine — is a row in its
  * head (François, 2026-09-17). Whatever the source, the files are ticked and
  * installed together by the bar over the list, and the browser stays open for
  * the next lot; a file on this machine is copied into the LoRA store. The
  * architecture card's LoRA section and a run configuration's default LoRAs
  * share it.
  */
class LoraInstallButton(
    browsers: BrowserServices,
    architecture: Architecture,
    service: LoraService,
    /** Also run when the button is clicked. */
    onOpen: () => Unit = () => ()
) extends Component {

  private val browsing = Var(false)

  /** What this architecture already holds, marked in the browser: finding out
    * by installing costs a download (François, 2026-09-17). Another
    * architecture's LoRAs are left out — installing the same file there is a
    * separate entity with its own files.
    */
  private val installed: Signal[Installed] =
    service.loras.map(all =>
      Installed.loras(all.filter(_.architectureId == architecture.id))
    )

  private def install(
      source: LoraInstallSource,
      grouping: LoraGrouping
  ): Unit =
    service.push(
      LoraService.Command.Install(architecture.id, source, grouping)
    )

  lazy val element: HtmlElement = span(
    button(
      cls := "button is-small is-info",
      span(cls := "plus-icon", "+"),
      " Add LoRA",
      title := s"Install a LoRA for ${architecture.label}",
      onClick --> { _ =>
        onOpen()
        browsing.set(true)
      }
    ),
    child <-- browsing.signal.map {
      case false => emptyNode
      case true  =>
        SourceBrowser(
          browsers = browsers,
          architecture = architecture,
          forLoras = true,
          // Nothing is picked file by file here: the ticked files install
          // through the bar, and the browser stays open for the next lot.
          onFile = (_, _) => (),
          onCancel = () => browsing.set(false),
          onInstall = Some(install),
          installed = installed
        ).element
    }
  )
}
