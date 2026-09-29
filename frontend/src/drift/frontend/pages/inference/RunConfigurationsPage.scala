package drift.frontend.pages.inference

import drift.frontend.Page
import drift.frontend.components.*
import drift.frontend.pages.sandbox.SandboxPage
import drift.frontend.services.*
import drift.frontend.services.RunConfigurationService.{Command, Event}
import drift.shared.*

import com.raquo.laminar.api.L.*

class RunConfigurationsPage(
    service: RunConfigurationService,
    cacheService: CacheService,
    downloadService: DownloadService,
    sessionService: SessionService,
    runtimeService: RuntimeService,
    assistantService: AssistantService,
    loraService: LoraService,
    /** Who each live session runs for (`specs/47-sandbox.md`). */
    projectService: ProjectService,
    /** The Civitai browser, for installing LoRAs from a configuration's form.
      */
    browsers: BrowserServices,
    /** Which runners' configurations this list covers. Both, since the
      * generation models and the assistants were merged into one page
      * (François, 2026-09-10): they are the same thing — a model, a runtime,
      * and a panel to try it in — and were only ever apart because the
      * assistant arrived second.
      */
    tools: List[RuntimeTool]
) extends Component {
  private val showForm = Var(false)

  private val searchQuery = Var("")
  private val tag = Var(Option.empty[String])

  private val toolArchitectures: Signal[List[Architecture]] =
    service.architectures.map(_.filter(a => tools.contains(a.tool)))

  /** The launch-time runtime pick per configuration. Deliberately not part of
    * `viewData`: the DOM select shows the pick the moment it is made, and this
    * only has to survive the card being rebuilt by other signals.
    */
  private val selectedRuntimeIds = Var(Map.empty[String, String])

  private val createModal = NewRunConfigurationModal(
    service,
    toolArchitectures,
    loraService,
    browsers,
    assistantService.library.ofKind(PromptKind.AssistantSystem),
    runtimeService.runtimes,
    onClose = () => showForm.set(false)
  )

  /** The configuration being edited, in the edit modal — creating and editing
    * both happen in a modal, never in place, so the grid never moves under the
    * user (François, 2026-09-15).
    */
  private val editCard = Var(Option.empty[RunConfigurationEditCard])

  /** What each configuration lacks before it can launch. Its effects are not
    * mounted: this page mounts the services it reads already.
    */
  private val prerequisites =
    LaunchPrerequisites(service, cacheService, downloadService, runtimeService)
  // Mirror of service.architectures kept in sync below, so event handlers
  // (startEdit) can read the current value synchronously.
  private val archsNow = Var(List.empty[Architecture])
  private val configurationsNow = Var(List.empty[RunConfiguration])

  private val viewData = Signal
    .combine(
      service.runConfigurations,
      service.architectures,
      // Paired, like the runtimes and their rules below: these two are one
      // filter between them.
      searchQuery.signal.combineWith(tag.signal)
    )
    .combineWith(
      service.allModels,
      cacheService.statuses,
      sessionService.sessions,
      // Paired rather than a tenth source: the rules arrive with the runtimes
      // in one response, and combining them here keeps the preview from
      // resolving against a rule list that has not landed yet.
      runtimeService.runtimes.combineWith(runtimeService.rules),
      // Paired too: a chat configuration's preview names every installed
      // LoRA adapter of its architecture (`specs/35-assistant-loras.md`).
      runtimeService.selection.combineWith(
        loraService.loras,
        // Who each live session runs for (`specs/47-sandbox.md`).
        projectService.projects
      )
    )
    .map {
      case (
            allConfigurations,
            archs,
            (search, wantedTag),
            models,
            cache,
            sessions,
            (runtimes, runtimeRules),
            (selection, loras, projects)
          ) =>
        val needle = search.trim.toLowerCase
        val configurations = allConfigurations.filter { configuration =>
          archs.find(_.id == configuration.architectureId) match {
            case None               => false
            case Some(architecture) =>
              tools.contains(architecture.tool) &&
              wantedTag.forall(architecture.tags.contains) &&
              // The architecture's name counts as the configuration's: "flux"
              // should find every configuration built on one.
              (needle.isEmpty ||
                configuration.label.toLowerCase.contains(needle) ||
                configuration.id.toLowerCase.contains(needle) ||
                architecture.label.toLowerCase.contains(needle))
          }
        }
        // The runtime a launch uses when none is picked, as the backend
        // resolves it (`specs/43`): the tool's default when it is the
        // runner's, else the runner's newest valid build.
        def runnerRuntime(
            tool: RuntimeTool,
            runner: RuntimeEngine
        ): Option[Runtime] = {
          val builds =
            runtimes.filter(r => r.tool == tool && r.engine == runner)
          builds
            .find(r => selection.defaultFor(tool).contains(r.id))
            .orElse(builds.filter(_.valid).sortBy(-_.createdAt).headOption)
        }
        // The launch-time choice: only runtimes of the architecture's engines
        // that can actually run, the configuration's runner's first and named
        // as such.
        def launchableRuntimes(
            tool: RuntimeTool,
            engines: List[RuntimeEngine],
            runner: RuntimeEngine
        ): List[Runtime] = {
          val valid = runtimes.filter(r =>
            r.valid && r.tool == tool && engines.contains(r.engine)
          )
          val chosen = runnerRuntime(tool, runner).map(_.id)
          val (first, rest) = valid.partition(r => chosen.contains(r.id))
          first.map(r => r.copy(label = s"${r.label} (its runner)")) ++
            rest.sortBy(_.label)
        }
        // Most recently used first. `lastUsedAt` starts as `createdAt`, so a
        // new configuration also lands on top. A launch in this very page view
        // bumps the live session's `startedAt` before the persisted
        // `lastUsedAt` is re-fetched, so the fresher of the two wins.
        val ordered = configurations.sortBy { rm =>
          -math.max(
            rm.lastUsedAt,
            sessions.get(rm.id).map(_.startedAt).getOrElse(0L)
          )
        }
        ordered.map { rm =>
          val arch = archs.find(_.id == rm.architectureId)
          val archName = arch.map(_.label).getOrElse(rm.architectureId)
          val tool = arch.map(_.tool).getOrElse(RuntimeTool.SdCpp)
          val engines = arch.fold(List(rm.runner))(_.runners)
          val runnerId = runnerRuntime(tool, rm.runner).map(_.id)
          val blockers = CommandLine.blockers(rm, archs, models, cache)
          // The preview resolves each weight to its cached path — the same
          // string the launcher passes (spec 07) — and falls back to naming
          // the source for weights not on disk yet.
          // Resolved against the runtime this card would launch on, so
          // the preview shows the argv the launcher will build — vetoed
          // flags dropped, and the reasons alongside
          // (`specs/16-parameter-resolution.md`).
          val previewRuntime = selectedRuntimeIds
            .now()
            .get(rm.id)
            .orElse(runnerId)
            .flatMap(id => runtimes.find(_.id == id))
          val rule = previewRuntime.flatMap(
            RuntimeRule.forRuntime(_, runtimeRules)
          )
          val resolved = CommandLine
            .resolve(
              rm,
              archs,
              models,
              m =>
                cache
                  .get(m.id)
                  .flatMap(_.path)
                  .getOrElse(sourceLabel(m.source)),
              // The launcher passes the absolute paths; the preview cannot
              // know the home directory, so it shows the `~` forms.
              loraModelDirectory = Some("~/.cache/drift/loras"),
              hiresUpscalersDirectory = Some("~/.cache/drift/upscale"),
              rule = rule,
              loraAdapters = LoraAdapters
                .relativePaths(rm.architectureId, loras)
                .map(path => s"~/.cache/drift/loras/$path")
            )
            .toOption
          val commandLine =
            resolved.map((arguments, _) =>
              CommandLine.render(arguments, tool.executableName)
            )
          val notes =
            resolved.map((_, notes) => notes).getOrElse(List.empty)
          RunConfigurationCard(
            rm,
            archName,
            blockers,
            commandLine,
            notes,
            sessions.get(rm.id),
            launchableRuntimes(tool, engines, rm.runner),
            selectedRuntimeIds
              .now()
              .get(rm.id)
              .orElse(runnerId),
            runtimeId => selectedRuntimeIds.update(_ + (rm.id -> runtimeId)),
            handleDelete,
            startEdit,
            prerequisites.of(Val(rm.id)),
            prerequisites,
            handleLaunch,
            id => sessionService.push(SessionService.Command.Stop(id)),
            sessions
              .get(rm.id)
              .map(session =>
                session.projectId.fold("the Sandbox")(id =>
                  projects.find(_.id == id).map(_.label).getOrElse("a project")
                )
              ),
            openSession
          ).element
        }
    }

  private def sourceLabel(source: ModelSource): String = source match {
    case HuggingFace(repo, file)   => s"<hf:$repo/$file>"
    case ModelScope(repo, file)    => s"<modelscope:$repo/$file>"
    case Civitai(id, _, fileId, _) => s"<civitai:$id/$fileId>"
    case Local(path)               => path
  }

  private def handleDelete(id: String): Unit =
    service.push(Command.Delete(id))

  /** An untouched select means "the default": no explicit pick is sent, so the
    * backend resolves whatever the default runtime is at spawn time.
    */
  private def handleLaunch(configurationId: String): Unit = {
    sessionService.push(
      SessionService.Command.Launch(
        configurationId,
        selectedRuntimeIds.now().get(configurationId),
        None
      )
    )
    openSandbox(configurationId)
  }

  /** Where a configuration is tried (`specs/47-sandbox.md`): the Sandbox, on
    * the kind its architecture makes.
    */
  private def openSandbox(configurationId: String): Unit =
    Page.Sandbox.navigateTo(
      configurationsNow
        .now()
        .find(_.id == configurationId)
        .flatMap(c => archsNow.now().find(_.id == c.architectureId))
        .map(SandboxPage.kindOf)
        .getOrElse(ProjectKind.Image)
    )

  /** A live session opens where it runs: its project, or the Sandbox. */
  private def openSession(session: Session): Unit =
    session.projectId match {
      case Some(projectId) => Page.ProjectWorkspace(projectId).navigate()
      case None            => openSandbox(session.runConfigurationId)
    }

  private def startEdit(rm: RunConfiguration): Unit = {
    val architecture = archsNow.now().find(_.id == rm.architectureId)
    editCard.set(
      Some(
        RunConfigurationEditCard(
          rm,
          architecture,
          service.modelService,
          loraService,
          browsers,
          assistantService.library.ofKind(PromptKind.AssistantSystem),
          runtimeService.runtimes
        )
      )
    )
  }

  private def cancelEdit(): Unit = {
    editCard.set(None)
  }

  private def handleSave(): Unit = {
    editCard.now().foreach { card =>
      service.push(Command.Update(card.rm.id, card.snapshot()))
    }
  }

  private def listElement: HtmlElement = div(
    div(
      cls := "level",
      div(
        cls := "level-left",
        h1(cls := "title text-primary", "Run configurations")
      ),
      div(
        cls := "level-right",
        button(
          cls := "button is-primary",
          span(cls := "plus-icon", "+"),
          " New Configuration",
          onClick --> (_ => showForm.set(true))
        )
      )
    ),
    div(
      cls := "list-filters mb-4",
      div(
        cls := "field is-expanded mb-0",
        div(
          cls := "control",
          input(
            cls := "input",
            placeholder := "Search run configurations\u2026",
            onInput.mapToValue --> searchQuery
          )
        )
      ),
      TagFilter(
        tag,
        // Only the tags of architectures this page lists, so a filter never
        // offers a kind with nothing behind it.
        service.architectures
          .map(_.filter(a => tools.contains(a.tool)))
          .map(TagFilter.inUse)
      )
    ),
    hr(),
    child <-- showForm.signal.map(if (_) createModal.element else emptyNode),
    child <-- editCard.signal.map {
      case None       => emptyNode
      case Some(card) =>
        NewRunConfigurationModal.frame(
          service,
          s"Edit: ${card.rm.label}",
          card.element,
          button(
            cls := "button is-success",
            "\uD83D\uDCBE Save",
            onClick --> (_ => handleSave())
          ),
          () => cancelEdit()
        )
    },
    div(
      cls := "run-configuration-grid",
      children <-- viewData
    )
  )

  lazy val element: HtmlElement = div(
    cls := "content",
    service.effects,
    cacheService.effects,
    downloadService.effects,
    sessionService.effects,
    runtimeService.effects,
    prerequisites.followInstalls,
    loraService.effects,
    projectService.effects,
    downloadService.events --> Observer {
      case DownloadService.Event.Finished(_) =>
        cacheService.push(CacheService.Command.Load)
        downloadService.push(DownloadService.Command.Load)
    },
    service.architectures --> archsNow,
    service.runConfigurations --> configurationsNow,
    service.events --> Observer {
      case Event.Updated(_) => cancelEdit()
      case _                => ()
    },
    // Each service loads its own entity, so all three have to be asked. Miss one
    // and the page renders with a silently empty list: without the model load,
    // landing here first (deep link, refresh) leaves every checkpoint dropdown
    // empty until the Architectures page happens to fetch the models.
    onMountCallback { _ =>
      service.push(Command.Load)
      service.modelService.push(ModelService.Command.Load)
      service.architectureService.push(ArchitectureService.Command.Load)
      cacheService.push(CacheService.Command.Load)
      sessionService.push(SessionService.Command.Load)
      runtimeService.push(RuntimeService.Command.Load)
      runtimeService.push(RuntimeService.Command.LoadInstalls)
      runtimeService.push(RuntimeService.Command.LoadInstallOptions)
      // The default LoRAs pick from the collection; without this it stayed
      // empty until the Architectures page had loaded it.
      loraService.push(LoraService.Command.Load)
      projectService.push(ProjectService.Command.Load)
    },
    ErrorBanner(service),
    ErrorBanner(service.modelService),
    ErrorBanner(service.architectureService),
    ErrorBanner(cacheService),
    ErrorBanner(downloadService),
    ErrorBanner(sessionService),
    ErrorBanner(runtimeService),
    ErrorBanner(loraService),
    listElement
  )
}
