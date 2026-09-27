package drift.frontend.pages.settings

import drift.frontend.components.Component
import drift.frontend.services.RuntimeService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The "ROCm (TheRock) build" field, shared by the install flows and the
  * runtime rows: it looks up the resolution for the gfx + ROCm version, renders
  * the picker over `pinned` / `onPick`, and asks the backend to resolve when
  * the pairing is not cached yet.
  */
class TheRockBuildField(
    runtimeService: RuntimeService,
    gfx: Signal[String],
    rocmVersion: Signal[Option[String]],
    /** The build pinned on purpose, or None for the resolver's choice. */
    pinned: Signal[Option[String]],
    onPick: Observer[Option[String]]
) extends Component {

  private def channelLabel(channel: TheRockChannel): String = channel match {
    case TheRockChannel.Stable           => "stable"
    case TheRockChannel.ReleaseCandidate => "release candidate"
    case TheRockChannel.Nightly          => "nightly"
  }

  private def choiceLabel(choice: TheRockChoice): String = {
    val marks = List(
      Option.when(choice.tagged)("tagged"),
      Option.when(choice.onDisk)("already downloaded")
    ).flatten
    val suffix = if (marks.isEmpty) "" else marks.mkString(" — ", ", ", "")
    s"${channelLabel(choice.channel)} ${choice.version}$suffix"
  }

  private def resolutionView(
      resolution: Option[TheRockResolution],
      pinnedVersion: Option[String]
  ): HtmlElement = resolution match {
    case None =>
      p(cls := "text-secondary is-size-7", "Resolving a matching ROCm build…")
    case Some(res) if res.alternatives.isEmpty =>
      p(
        cls := "has-text-danger is-size-7",
        s"No tagged stable ROCm build is published for ${res.gfxTarget}."
      )
    case Some(res) =>
      val autoChoice = res.chosen
      div(
        select(
          cls := "select",
          onChange.mapToValue --> Observer[String] { value =>
            onPick.onNext(Option.when(value.nonEmpty)(value))
          },
          option(
            value := "",
            if (pinnedVersion.isEmpty) selected := true else emptyNode,
            autoChoice match {
              case Some(choice) =>
                s"Automatic — ${choiceLabel(choice)}"
              case None => "Automatic"
            }
          ),
          res.alternatives.map(choice =>
            option(
              value := choice.version,
              if (pinnedVersion.contains(choice.version)) selected := true
              else emptyNode,
              choiceLabel(choice)
            )
          )
        ),
        p(
          cls := "text-secondary is-size-7 mt-1",
          s"The release declares ROCm ${res.neededVersion}. " +
            (autoChoice match {
              case Some(choice) if choice.tagged =>
                s"Using tagged stable release ${choice.version}. Pick a " +
                  "different one below to test (e.g. an older release)."
              case Some(choice) if choice.onDisk =>
                s"Reusing ${choice.version} from disk — no matching tagged " +
                  "release. Pick another below to test."
              case Some(_) => "Pick a build below."
              case None    =>
                "No tagged stable release matches this version — pick one " +
                  "below to test, even if the version differs."
            })
        )
      )
  }

  lazy val element: HtmlElement =
    div(
      cls := "field",
      label(cls := "label is-small text-secondary", "ROCm (TheRock) build"),
      // Resolve whenever the (gfx, ROCm version) pair is set and not yet cached.
      // Bound as a Signal (not `.changes`) so the initial value fires on mount;
      // once the result lands the guard makes this None, so it never loops.
      gfx
        .combineWith(rocmVersion, runtimeService.theRockResolutions)
        .map {
          case (gfx, Some(rocm), resolutions)
              if !resolutions.contains(RuntimeService.theRockKey(gfx, rocm)) =>
            Some((gfx, rocm))
          case _ => None
        } --> Observer[Option[(String, String)]] {
        case Some((gfx, rocm)) =>
          runtimeService.push(RuntimeService.Command.ResolveTheRock(gfx, rocm))
        case None => ()
      },
      child <-- gfx
        .combineWith(
          rocmVersion,
          runtimeService.theRockResolutions,
          pinned
        )
        .map {
          case (gfx, Some(rocm), resolutions, pinnedVersion) =>
            resolutionView(
              resolutions.get(RuntimeService.theRockKey(gfx, rocm)),
              pinnedVersion
            )
          case _ => emptyNode
        }
    )
}
