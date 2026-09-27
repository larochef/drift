package drift.frontend.pages.settings

import drift.frontend.components.Component
import drift.frontend.services.RuntimeService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Installing a runtime from a release: pick a tool, then either the newest
  * release for a backend — registered as a "latest" runtime — or a specific
  * release and build. A ROCm build also takes a GPU target and a TheRock build.
  * Its fields live as long as the section, so switching tabs or reopening the
  * modal keeps them.
  */
class InstallRuntimeForm(
    runtimeService: RuntimeService,
    /** Strix Halo by default; shared with the runtime rows. */
    gfxTarget: Var[String],
    /** Closes the modal once an install is started. */
    onDone: () => Unit
) extends Component {
  import RuntimeService.Command
  import RuntimeOptions.*

  /** Which tool the install forms are about; both draw on its releases. */
  private val installTool = Var[RuntimeTool](RuntimeTool.SdCpp)
  private val selectedRelease = Var(Option.empty[RuntimeRelease])
  private val selectedAsset = Var(Option.empty[RuntimeReleaseAsset])

  private val installReleases: Signal[List[RuntimeRelease]] =
    installTool.signal
      .combineWith(runtimeService.releases)
      .map((tool, all) => all.getOrElse(tool, Nil))

  /** Backend for the "install latest" shortcut — Vulkan by default, the one
    * confirmed working on the target machine (see bugs/17).
    */
  private val latestBackend = Var[RuntimeBackend](RuntimeBackend.Vulkan)

  /** An explicit TheRock version to pair, or None for automatic resolution. One
    * per install flow — the specific-release form and the install-latest
    * shortcut each keep their own so their pickers do not fight.
    */
  private val selectedTheRockVersion = Var(Option.empty[String])
  private val latestTheRockVersion = Var(Option.empty[String])

  private def buildField(
      rocmVersion: Signal[Option[String]],
      pinned: Var[Option[String]]
  ): HtmlElement =
    TheRockBuildField(
      runtimeService,
      gfxTarget.signal,
      rocmVersion,
      pinned.signal,
      pinned.writer
    ).element

  /** The one-click path: pick a backend, drift resolves and installs the newest
    * release of the chosen tool for it under `latest-<backend>`, and the
    * "Upgrade" button on that runtime keeps it current later.
    */
  private def installLatestSection: HtmlElement = div(
    cls := "box bg-card p-3 mb-4",
    p(
      cls := "text-secondary is-size-7 mb-2",
      child.text <-- installTool.signal.map(tool =>
        s"Install the newest ${tool.displayName} release for a backend. It is registered as a " +
          "\"latest\" runtime you can upgrade in place when a newer release lands, " +
          "rather than pinned to today's tag."
      )
    ),
    div(
      cls := "field is-grouped is-align-items-center",
      div(
        cls := "control",
        select(
          cls := "select is-small",
          onChange.mapToValue --> Observer[String] { name =>
            RuntimeBackend.values
              .find(_.toString == name)
              .foreach(latestBackend.set)
            latestTheRockVersion.set(None)
          },
          RuntimeBackend.values.toList.map(backend =>
            option(
              value := backend.toString,
              if (backend == latestBackend.now()) selected := true
              else emptyNode,
              backendLabel(backend)
            )
          )
        )
      ),
      child <-- latestBackend.signal.map {
        case RuntimeBackend.Rocm =>
          div(
            cls := "control",
            child <-- runtimeService.targets
              .combineWith(gfxTarget.signal)
              .map { (targets, gfx) =>
                val known = if (targets.isEmpty) List(gfx) else targets
                select(
                  cls := "select is-small",
                  onChange.mapToValue --> Observer[String] { target =>
                    gfxTarget.set(target)
                    latestTheRockVersion.set(None)
                  },
                  known.map(target =>
                    option(
                      value := target,
                      if (target == gfx) selected := true else emptyNode,
                      target
                    )
                  )
                )
              }
          )
        case _ => emptyNode
      },
      div(
        cls := "control",
        child <-- latestBackend.signal
          .combineWith(installReleases)
          .map { (backend, releases) =>
            val newest = newestTagFor(releases, backend)
            button(
              cls := "button is-primary is-small",
              disabled := newest.isEmpty,
              "Install latest",
              onClick --> { _ =>
                onDone()
                val rocm = backend == RuntimeBackend.Rocm
                runtimeService.push(
                  Command.InstallLatest(
                    InstallLatestRequest(
                      tool = installTool.now(),
                      backend = backend,
                      gfxTarget = Option.when(rocm)(gfxTarget.now()),
                      theRockVersion =
                        if (rocm) latestTheRockVersion.now() else None
                    )
                  )
                )
              }
            )
          }
      )
    ),
    // ROCm installs get the same TheRock build picker as the specific-release
    // form, resolved against the newest ROCm release's ROCm version.
    child <-- latestBackend.signal.map {
      case RuntimeBackend.Rocm =>
        buildField(
          installReleases.map(newestRocmVersion),
          latestTheRockVersion
        )
      case _ => emptyNode
    }
  )

  /** ROCm needs a GPU target and a matching TheRock ROCm build. drift resolves
    * the build — an already-downloaded one first, then the most stable channel
    * that matches the asset's ROCm version — and shows it here, with a dropdown
    * to pin a different build on purpose.
    */
  private def rocmChooser: HtmlElement = div(
    gpuTargetField(selectedTheRockVersion),
    buildField(
      selectedAsset.signal.map(_.flatMap(_.rocmVersion)),
      selectedTheRockVersion
    )
  )

  /** The GPU-target select. Changing it drops any pin, since a different GPU
    * has a different set of builds; the shared build field re-resolves on its
    * own.
    */
  private def gpuTargetField(selVar: Var[Option[String]]): HtmlElement =
    div(
      cls := "field",
      label(cls := "label is-small text-secondary", "GPU target"),
      child <-- runtimeService.targets
        .combineWith(gfxTarget.signal)
        .map { (targets, gfx) =>
          val known = if (targets.isEmpty) List(gfx) else targets
          select(
            cls := "select",
            onChange.mapToValue --> Observer[String] { target =>
              gfxTarget.set(target)
              selVar.set(None)
            },
            known.map(target =>
              option(
                value := target,
                if (target == gfx) selected := true else emptyNode,
                target
              )
            )
          )
        }
    )

  lazy val element: HtmlElement = div(
    div(
      cls := "field",
      label(cls := "label is-small text-secondary", "Tool"),
      toolSelect(
        installTool,
        () => {
          selectedRelease.set(None)
          selectedAsset.set(None)
          selectedTheRockVersion.set(None)
          latestTheRockVersion.set(None)
        }
      )
    ),
    installLatestSection,
    p(
      cls := "label is-small text-secondary",
      "…or pick a specific release"
    ),
    div(
      cls := "field",
      label(
        cls := "label is-small text-secondary",
        child.text <-- installTool.signal.map(t => s"${t.displayName} release")
      ),
      child <-- installReleases
        .combineWith(selectedRelease.signal)
        .map { (releases, chosenRelease) =>
          if (releases.isEmpty)
            p(
              cls := "text-secondary is-size-7",
              "Loading releases from GitHub…"
            )
          else
            select(
              cls := "select",
              onChange.mapToValue --> Observer[String] { tag =>
                selectedRelease.set(releases.find(_.tag == tag))
                selectedAsset.set(None)
                selectedTheRockVersion.set(None)
              },
              option(value := "", "Choose a release"),
              releases.map(release =>
                option(
                  value := release.tag,
                  if (chosenRelease.exists(_.tag == release.tag))
                    selected := true
                  else emptyNode,
                  s"${release.tag}  (${release.publishedAt.take(10)})"
                )
              )
            )
        }
    ),
    child <-- selectedRelease.signal
      .combineWith(selectedAsset.signal)
      .map {
        case (Some(release), chosen) =>
          div(
            cls := "field",
            label(cls := "label is-small text-secondary", "Build"),
            release.assets.map { asset =>
              val text = (List(
                backendLabel(asset.backend),
                LaunchBlocker.humanBytes(asset.sizeBytes)
              ) ++ asset.rocmVersion.map(v => s"needs ROCm $v"))
                .mkString(" — ")
              label(
                cls := "radio is-block",
                input(
                  typ := "radio",
                  nameAttr := "runtime-asset",
                  checked := chosen.exists(_.name == asset.name),
                  onChange --> { _ =>
                    selectedAsset.set(Some(asset))
                    // A new asset means a new ROCm requirement; drop any pin so
                    // the resolver's preferred choice is used again.
                    selectedTheRockVersion.set(None)
                    asset.rocmVersion.foreach(rocm =>
                      runtimeService.push(
                        RuntimeService.Command
                          .ResolveTheRock(gfxTarget.now(), rocm)
                      )
                    )
                  }
                ),
                s" $text"
              )
            }
          )
        case _ => emptyNode
      },
    child <-- selectedAsset.signal.map {
      case Some(asset) if asset.backend == RuntimeBackend.Rocm =>
        rocmChooser
      case _ => emptyNode
    },
    div(
      cls := "field mt-2",
      button(
        cls := "button is-primary",
        "Install runtime",
        disabled <-- selectedAsset.signal.map(_.isEmpty),
        onClick --> { _ =>
          onDone()
          (selectedRelease.now(), selectedAsset.now()) match {
            case (Some(release), Some(asset)) =>
              val rocm = asset.backend == RuntimeBackend.Rocm
              runtimeService.push(
                Command.Install(
                  InstallRuntimeRequest(
                    tool = installTool.now(),
                    releaseTag = release.tag,
                    asset = asset,
                    gfxTarget = Option.when(rocm)(gfxTarget.now()),
                    theRockVersion =
                      if (rocm) selectedTheRockVersion.now() else None
                  )
                )
              )
            case _ => ()
          }
        }
      )
    )
  )
}
