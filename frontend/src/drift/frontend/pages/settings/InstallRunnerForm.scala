package drift.frontend.pages.settings

import drift.frontend.components.Component
import drift.frontend.services.RuntimeService

import com.raquo.laminar.api.L.*

/** Installing drift's own runner (`specs/43`): it ships inside drift, so the
  * only choice is the TheRock build, paired against the ROCm its kernels were
  * built with. One install gives both its runtimes, chat and images.
  */
class InstallRunnerForm(
    runtimeService: RuntimeService,
    /** Closes the modal once the install is started. */
    onDone: () => Unit
) extends Component {
  import RuntimeService.Command

  /** The runner's kernels are built for Strix Halo only. */
  private val Gfx = "gfx1151"

  private val pinned = Var(Option.empty[String])

  private def install: HtmlElement = div(
    p(
      cls := "text-secondary is-size-7 mb-3",
      "drift's own engine for Strix Halo (gfx1151). It ships inside drift, so " +
        "nothing is downloaded but the ROCm build it runs on. It installs as two " +
        "runtimes, a llama.cpp one for chat and an sd-cpp one for images, and " +
        "runs the architectures that list it as a runner."
    ),
    TheRockBuildField(
      runtimeService,
      Val(Gfx),
      runtimeService.runnerOffer.map(_.flatMap(_.rocmVersion)),
      pinned.signal,
      pinned.writer
    ).element,
    button(
      cls := "button is-primary",
      "Install the drift runner",
      onClick --> (_ => {
        runtimeService.push(Command.InstallRunner(pinned.now()))
        onDone()
      })
    )
  )

  lazy val element: HtmlElement = div(
    onMountCallback(_ => runtimeService.push(Command.LoadRunnerOffer)),
    child <-- runtimeService.runnerOffer.map {
      case None => p(cls := "text-secondary is-size-7", "Asking drift…")
      case Some(offer) if !offer.available =>
        p(
          cls := "text-secondary is-size-7",
          s"The drift runner cannot be installed: ${offer.reason.getOrElse("unknown reason")}."
        )
      case Some(_) => install
    }
  )
}
