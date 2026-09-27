package drift.frontend.pages.settings

import drift.frontend.components.Component
import drift.frontend.services.RuntimeService
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** One registered runtime: its tool, validation state and pairing, whether it
  * is its tool's default or follows the newest release, and its actions — set
  * default, upgrade (with the TheRock pick when the newest release declares
  * another ROCm), change ROCm, revalidate, delete.
  */
class RuntimeRow(
    runtime: Runtime,
    selection: RuntimeSelection,
    releases: Map[RuntimeTool, List[RuntimeRelease]],
    runtimeService: RuntimeService,
    /** Per-runtime TheRock picks, for a "Change ROCm" or an upgrade whose
      * release declares another ROCm. The section's, so a re-rendered list
      * keeps them.
      */
    theRockPicks: Var[Map[String, Option[String]]],
    /** Which rows have the change panel open — the section's, for the same
      * reason.
      */
    changeRocmOpen: Var[Set[String]],
    /** The GPU target resolved for a runtime that names none. */
    gfxTarget: Var[String]
) extends Component {
  import RuntimeService.Command
  import RuntimeOptions.{backendLabel, newestAssetFor}

  lazy val element: HtmlElement = {
    val isDefault = selection.defaultFor(runtime.tool).contains(runtime.id)
    val newest =
      newestAssetFor(releases.getOrElse(runtime.tool, Nil), runtime.backend)
    val newestTag = newest.map(_._1)
    val updateAvailable =
      runtime.tracksLatest && newestTag.exists(_ != runtime.releaseTag)
    // The newest release declares a ROCm version the runtime is not paired
    // with: the upgrade then offers the same select an install shows, instead
    // of re-pairing silently (`specs/17-assistant-runtime.md`).
    val declaredNow = newest.flatMap(_._2.rocmVersion)
    val pairingChanges =
      runtime.backend == RuntimeBackend.Rocm && updateAvailable &&
        declaredNow.exists(now =>
          !runtime.rocmVersion.exists(RocmVersions.satisfies(_, now))
        )
    val pairingLabel = (runtime.rocmVersion, runtime.theRockVersion) match {
      case (Some(declared), Some(paired))
          if RocmVersions.satisfies(declared, paired) =>
        Some(s"TheRock $paired (declared)")
      case (Some(declared), Some(paired)) =>
        Some(s"TheRock $paired (chosen — release declares $declared)")
      case (_, Some(paired)) => Some(s"TheRock $paired")
      case _                 => None
    }
    val pick: Signal[Option[String]] =
      theRockPicks.signal.map(_.getOrElse(runtime.id, None))
    val onPick = Observer[Option[String]](version =>
      theRockPicks.update(_ + (runtime.id -> version))
    )
    val rowGfx = Val(runtime.gfxTarget.getOrElse(gfxTarget.now()))
    val canChangeRocm =
      runtime.backend == RuntimeBackend.Rocm && !runtime.adopted
    def buildField(rocm: Option[String]): HtmlElement =
      TheRockBuildField(runtimeService, rowGfx, Val(rocm), pick, onPick).element
    div(
      cls := "box bg-card p-3 mb-2",
      div(
        cls := "level is-mobile is-marginless",
        div(
          cls := "level-left",
          div(
            p(
              cls := "text-primary mb-1",
              strong(runtime.label),
              span(cls := "tag is-small ml-2", runtime.tool.displayName),
              if (runtime.engine == RuntimeEngine.DriftRunner)
                span(
                  cls := "tag is-primary is-light is-small ml-2",
                  title := runtime.modelKinds.fold("")(kinds =>
                    s"Runs ${kinds.mkString(", ")}"
                  ),
                  "drift runner"
                )
              else emptyNode,
              if (isDefault)
                span(
                  cls := "tag is-info is-small ml-2",
                  title := s"The ${runtime.tool.displayName} default",
                  "default"
                )
              else emptyNode,
              if (runtime.adopted)
                span(cls := "tag is-light is-small ml-2", "adopted")
              else emptyNode,
              if (runtime.tracksLatest)
                span(
                  cls := "tag is-link is-small ml-2",
                  title :=
                    s"Follows the newest ${runtime.tool.displayName} release",
                  "latest"
                )
              else emptyNode,
              if (updateAvailable)
                span(
                  cls := "tag is-warning is-small ml-2",
                  title := newestTag
                    .map(t => s"newer release available: $t")
                    .getOrElse(""),
                  "update available"
                )
              else emptyNode,
              if (runtime.valid)
                span(
                  cls := "tag is-success is-small ml-2",
                  title := runtime.reportedVersion.getOrElse(""),
                  runtime.reportedVersion.getOrElse("valid")
                )
              else
                span(
                  cls := "tag is-danger is-small ml-2",
                  title := runtime.validationError.getOrElse(""),
                  "invalid"
                )
            ),
            p(
              cls := "text-secondary is-size-7 mb-1",
              (List(backendLabel(runtime.backend), runtime.releaseTag) ++
                runtime.rocmVersion.map(v => s"ROCm $v") ++
                runtime.gfxTarget ++
                pairingLabel).mkString(" · ")
            ),
            p(cls := "text-secondary is-size-7", runtime.installedAt),
            runtime.validationError match {
              case Some(error) =>
                p(cls := "has-text-danger is-size-7", error)
              case None => emptyNode
            }
          )
        ),
        div(
          cls := "level-right",
          if (!isDefault)
            button(
              cls := "button is-small mr-1",
              disabled := !runtime.valid,
              title :=
                (if (runtime.valid) ""
                 else "An invalid runtime cannot be the default"),
              "Set default",
              onClick --> { _ =>
                runtimeService.push(
                  Command.SetDefault(runtime.tool, Some(runtime.id))
                )
              }
            )
          else emptyNode,
          if (runtime.tracksLatest)
            button(
              cls := (if (updateAvailable) "button is-primary is-small mr-1"
                      else "button is-small mr-1"),
              title := newestTag
                .map(t =>
                  if (pairingChanges)
                    s"Upgrade to $t (ROCm ${declaredNow.getOrElse("?")}, paired ${runtime.theRockVersion
                        .getOrElse("nothing")})"
                  else if (updateAvailable) s"Upgrade to $t"
                  else s"Already on the latest release ($t)"
                )
                .getOrElse("Check for a newer release"),
              if (updateAvailable) "⬆ Upgrade" else "Check for update",
              onClick --> { _ =>
                val pairing =
                  if (pairingChanges)
                    theRockPicks.now().getOrElse(runtime.id, None)
                  else None
                runtimeService.push(Command.Upgrade(runtime.id, pairing))
              }
            )
          else emptyNode,
          if (canChangeRocm)
            button(
              cls := "button is-small mr-1",
              title := "Pair this release with another TheRock build",
              "Change ROCm",
              onClick --> { _ =>
                changeRocmOpen.update(open =>
                  if (open.contains(runtime.id)) open - runtime.id
                  else open + runtime.id
                )
              }
            )
          else emptyNode,
          button(
            cls := "button is-small mr-1",
            "Revalidate",
            onClick --> (_ => runtimeService.push(Command.Validate(runtime.id)))
          ),
          button(
            cls := "button is-danger is-small",
            "🗑️",
            title := "Delete runtime",
            onClick --> { _ =>
              val filesNote =
                if (runtime.adopted)
                  "The adopted files stay where they are."
                else "Its files under ~/.cache/drift/runtimes are deleted too."
              if (window.confirm(s"Delete ${runtime.label}?\n\n$filesNote"))
                runtimeService.push(Command.Delete(runtime.id))
            }
          )
        )
      ),
      if (pairingChanges)
        div(
          cls := "mt-2",
          p(
            cls := "is-size-7 has-text-warning mb-1",
            s"The newest release (${newestTag.getOrElse("?")}) declares ROCm ${declaredNow
                .getOrElse("?")}; this runtime is paired with ${runtime.theRockVersion
                .getOrElse("nothing")}. The upgrade pairs the build chosen here:"
          ),
          buildField(declaredNow)
        )
      else emptyNode,
      if (canChangeRocm)
        child <-- changeRocmOpen.signal.map(_.contains(runtime.id)).map {
          case false => emptyNode
          case true  =>
            val key = runtime.rocmVersion
              .map(RuntimeService.theRockKey(rowGfx.now(), _))
            div(
              cls := "mt-2",
              buildField(runtime.rocmVersion),
              child <-- pick
                .combineWith(runtimeService.theRockResolutions)
                .map { (pinned, resolutions) =>
                  // "Automatic" means the resolver's choice for the declared
                  // version — the same build an install would pair.
                  val target = pinned.orElse(
                    key
                      .flatMap(resolutions.get)
                      .flatMap(_.chosen.map(_.version))
                  )
                  div(
                    cls := "buttons are-small",
                    button(
                      cls := "button is-primary is-small",
                      disabled := target.isEmpty ||
                        target == runtime.theRockVersion,
                      title := target
                        .map(v =>
                          if (runtime.theRockVersion.contains(v))
                            s"Already paired with $v"
                          else s"Re-pair with TheRock $v"
                        )
                        .getOrElse("Pick a build first"),
                      target
                        .map(v => s"Re-pair with $v")
                        .getOrElse("Re-pair"),
                      onClick --> { _ =>
                        target.foreach { version =>
                          runtimeService.push(
                            Command.ChangeTheRock(runtime.id, version)
                          )
                          changeRocmOpen.update(_ - runtime.id)
                        }
                      }
                    ),
                    button(
                      cls := "button is-small",
                      "Cancel",
                      onClick --> (_ => changeRocmOpen.update(_ - runtime.id))
                    )
                  )
                }
            )
        }
      else emptyNode
    )
  }
}
