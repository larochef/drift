package drift.frontend.pages.settings

import drift.frontend.components.Component
import drift.frontend.services.RuntimeService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Registering a release that is already unpacked on disk, without downloading
  * anything; the directory is validated by running its executable. Its fields
  * live as long as the section, so switching tabs or reopening the modal keeps
  * them.
  */
class AdoptRuntimeForm(
    runtimeService: RuntimeService,
    /** Closes the modal once the runtime is registered. */
    onDone: () => Unit
) extends Component {
  import RuntimeOptions.{backendLabel, toolSelect}

  private val adoptPath = Var("")
  private val adoptLabel = Var("")
  private val adoptTool = Var[RuntimeTool](RuntimeTool.SdCpp)
  private val adoptBackend = Var[RuntimeBackend](RuntimeBackend.Rocm)
  private val adoptTheRockPath = Var("")

  private def register(): Unit = {
    onDone()
    val path = adoptPath.now().trim
    if (path.nonEmpty) {
      val fallback = path
        .split('/')
        .filter(_.nonEmpty)
        .lastOption
        .getOrElse("adopted")
      val label =
        Some(adoptLabel.now().trim).filter(_.nonEmpty).getOrElse(fallback)
      val id = label.toLowerCase
        .map(c => if (c.isLetterOrDigit) c else '-')
        .split('-')
        .filter(_.nonEmpty)
        .mkString("-")
      runtimeService.push(
        RuntimeService.Command.Register(
          Runtime(
            id = id,
            label = label,
            tool = adoptTool.now(),
            backend = adoptBackend.now(),
            // Unknown for a directory drift did not fetch; validation
            // stores the version the binary itself reports.
            releaseTag = "unknown",
            installedAt = path,
            theRockPath = Some(adoptTheRockPath.now().trim).filter(_.nonEmpty),
            adopted = true,
            createdAt = System.currentTimeMillis().toLong,
            modelKinds = None
          )
        )
      )
      adoptPath.set("")
      adoptLabel.set("")
      adoptTheRockPath.set("")
    }
  }

  lazy val element: HtmlElement = div(
    p(
      cls := "text-secondary is-size-7",
      child.text <-- adoptTool.signal.map(tool =>
        s"Register an already-unpacked ${tool.displayName} release (for example lemonade's " +
          "/var/cache/lemonade/bin/sd-cpp/rocm-stable) without downloading anything. " +
          s"The directory is validated by running its ${tool.executableName}."
      )
    ),
    div(
      cls := "field",
      label(cls := "label is-small text-secondary", "Tool"),
      toolSelect(adoptTool, () => ())
    ),
    div(
      cls := "field",
      label(cls := "label is-small text-secondary", "Directory"),
      input(
        cls := "input",
        placeholder := "/var/cache/lemonade/bin/sd-cpp/rocm-stable",
        controlled(
          value <-- adoptPath.signal,
          onInput.mapToValue --> adoptPath
        )
      )
    ),
    div(
      cls := "field",
      label(cls := "label is-small text-secondary", "Backend"),
      select(
        cls := "select",
        onChange.mapToValue --> Observer[String] { name =>
          RuntimeBackend.values
            .find(_.toString == name)
            .foreach(adoptBackend.set)
        },
        RuntimeBackend.values.toList.map(backend =>
          option(
            value := backend.toString,
            if (backend == adoptBackend.now()) selected := true else emptyNode,
            backendLabel(backend)
          )
        )
      )
    ),
    child <-- adoptBackend.signal.map {
      case RuntimeBackend.Rocm =>
        div(
          cls := "field",
          label(
            cls := "label is-small text-secondary",
            "TheRock directory (optional — its lib goes on LD_LIBRARY_PATH)"
          ),
          input(
            cls := "input",
            placeholder := "/path/to/therock-dist",
            controlled(
              value <-- adoptTheRockPath.signal,
              onInput.mapToValue --> adoptTheRockPath
            )
          )
        )
      case _ => emptyNode
    },
    div(
      cls := "field",
      label(cls := "label is-small text-secondary", "Label"),
      input(
        cls := "input",
        placeholder := "lemonade rocm-stable",
        controlled(
          value <-- adoptLabel.signal,
          onInput.mapToValue --> adoptLabel
        )
      )
    ),
    div(
      cls := "field",
      button(
        cls := "button is-primary",
        "Register runtime",
        disabled <-- adoptPath.signal.map(_.trim.isEmpty),
        onClick --> (_ => register())
      )
    )
  )
}
