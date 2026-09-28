package drift.frontend.components

import drift.frontend.services.LaunchPrerequisites
import drift.frontend.services.LaunchPrerequisites.*

import com.raquo.laminar.api.L.*

/** A button that starts a model, replaced while the model cannot start
  * (`specs/46-starter-configurations.md`): a notice that no runtime of its tool
  * is installed, with the button that installs one, and the download of its
  * missing weights. The launch comes back by itself once both are there.
  */
class LaunchOrDownload(
    missing: Signal[Option[Missing]],
    prerequisites: LaunchPrerequisites,
    launch: HtmlElement
) extends Component {

  private def weightsControl(pending: PendingWeights): HtmlElement =
    pending match {
      case PendingWeights(Nil, _) =>
        button(
          cls := "button is-small is-info",
          disabled := true,
          "⬇ Downloading the weights…"
        )
      case PendingWeights(idle, _) =>
        button(
          cls := "button is-small is-info",
          title := "The weights are not on disk yet: download them first",
          s"⬇ Download the weights (${idle.size})",
          onClick --> (_ => prerequisites.download(idle))
        )
    }

  lazy val element: HtmlElement = span(
    child <-- missing.map {
      case None          => launch
      case Some(missing) =>
        span(
          cls := "launch-prerequisites",
          missing.runtime.map(LaunchOrDownload.runtimeNotice(_, prerequisites)),
          missing.weights.map(weightsControl)
        )
    }
  )
}

object LaunchOrDownload {

  /** The notice that no runtime of the configuration's engine is installed,
    * with the choice of build — the recommended one preselected, drift's runner
    * among them where the architecture runs on it — and the button that
    * installs it, switching the configuration when the pick is another engine.
    * Also shown on its own where a picker, not a button, starts models.
    */
  def runtimeNotice(
      need: RuntimeNeed,
      prerequisites: LaunchPrerequisites
  ): HtmlElement = {
    val chosen = Var(need.options.headOption)
    span(
      cls := "launch-prerequisite",
      span(
        cls := "is-size-7 text-secondary",
        if (need.installing)
          s"Installing the ${need.use} runtime…"
        else if (need.installed.isEmpty)
          s"No ${need.use} runtime is installed."
        // One is, but not one these run on: name the models it is for.
        else
          s"Choose the ${need.use} runtime for " +
            need.configurations.map(_._1.label).mkString(", ")
      ),
      need.failure
        .filterNot(_ => need.installing)
        .map(reason =>
          span(
            cls := "is-size-7 has-text-danger",
            title := reason,
            s"The last install failed: ${reason.linesIterator.nextOption().getOrElse(reason)}"
          )
        ),
      if (need.installing)
        // Bulma's spinner: the install shows its progress in the downloads
        // panel, and the notice goes once it has validated.
        button(
          cls := "button is-small is-info is-loading",
          disabled := true,
          "Installing"
        )
      else if (need.options.isEmpty)
        span(cls := "is-size-7 text-secondary", "Install one from Settings.")
      else
        span(
          cls := "launch-prerequisite",
          span(
            cls := "select is-small",
            select(
              title := "The build to install; Settings offers other releases",
              need.options.map(option =>
                com.raquo.laminar.api.L.option(
                  value := option.runtimeId,
                  // The first is the preselected one (`preference`).
                  defaultSelected := need.options.headOption.contains(option),
                  option.label +
                    (if (need.options.headOption.contains(option))
                       " (recommended)"
                     else "") +
                    (if (need.installed.contains(option.engine)) " (installed)"
                     else "")
                )
              ),
              onChange.mapToValue --> (id =>
                chosen.set(need.options.find(_.runtimeId == id))
              )
            )
          ),
          button(
            cls := "button is-small is-info",
            title <-- chosen.signal.map(
              _.filter(need.switches).fold(
                "Installs the newest release; it becomes the default runtime."
              )(option =>
                s"Runs on ${option.engine.displayName} from now on: the " +
                  "configuration's runner changes (edit it to switch back)."
              )
            ),
            child.text <-- chosen.signal.map {
              case Some(option) if need.installed.contains(option.engine) =>
                "Use it"
              case _ if need.failure.isDefined => "⬇ Try again"
              case _                           => "⬇ Install"
            },
            onClick --> (_ =>
              chosen.now().foreach(prerequisites.choose(need, _))
            )
          )
        )
    )
  }
}
