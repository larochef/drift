package drift.frontend.pages.settings

import drift.shared.*

import com.raquo.laminar.api.L.*

/** What the runtime rows and forms share: labels, the tool select, and the
  * newest release per backend.
  */
object RuntimeOptions {

  def backendLabel(backend: RuntimeBackend): String = backend match {
    case RuntimeBackend.Rocm   => "ROCm"
    case RuntimeBackend.Vulkan => "Vulkan"
    case RuntimeBackend.Cpu    => "CPU"
  }

  /** A tool select, shared by the install and adopt forms. */
  def toolSelect(
      selVar: Var[RuntimeTool],
      onSwitch: () => Unit
  ): HtmlElement =
    select(
      cls := "select is-small",
      onChange.mapToValue --> Observer[String] { name =>
        RuntimeTool.values.find(_.toString == name).foreach { tool =>
          selVar.set(tool)
          onSwitch()
        }
      },
      RuntimeTool.values.toList.map(tool =>
        option(
          value := tool.toString,
          selected <-- selVar.signal.map(_ == tool),
          tool.displayName
        )
      )
    )

  /** The ROCm version the newest ROCm release declares, which the
    * install-latest shortcut must resolve a TheRock build against.
    */
  def newestRocmVersion(releases: List[RuntimeRelease]): Option[String] =
    releases.iterator
      .flatMap(_.assets.find(_.backend == RuntimeBackend.Rocm))
      .flatMap(_.rocmVersion)
      .nextOption()

  /** The newest release tag carrying an asset for a backend, or None while the
    * listing is still loading. Releases arrive newest first, so the first match
    * is the newest.
    */
  def newestTagFor(
      releases: List[RuntimeRelease],
      backend: RuntimeBackend
  ): Option[String] = newestAssetFor(releases, backend).map(_._1)

  def newestAssetFor(
      releases: List[RuntimeRelease],
      backend: RuntimeBackend
  ): Option[(String, RuntimeReleaseAsset)] =
    releases.iterator
      .flatMap(release =>
        release.assets.find(_.backend == backend).map(a => (release.tag, a))
      )
      .nextOption()
}
