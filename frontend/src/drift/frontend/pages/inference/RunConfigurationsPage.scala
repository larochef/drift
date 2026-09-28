package drift.frontend.pages.inference

import drift.frontend.components.*
import drift.frontend.pages.assistant.AssistantPanel
import drift.frontend.pages.generate.GenerationPanel
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
    generationService: GenerationService,
    assistantService: AssistantService,
    loraService: LoraService,
    /** The projects a free-play result can be kept into (`specs/22-…`). */
    projectService: ProjectService,
    /** The session's own output (`specs/13-log-streaming.md`). */
    logService: LogService,
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

  /** Whether to show the list even though something is live. */
  private val showList = Var(false)

  private val searchQuery = Var("")
  private val tag = Var(Option.empty[String])

  private val toolArchitectures: Signal[List[Architecture]] =
    service.architectures.map(_.filter(a => tools.contains(a.tool)))

  /** The launch-time runtime pick per configuration. Deliberately not part of
    * `viewData`: the DOM select shows the pick the moment it is made, and this
    * only has to survive the card being rebuilt by other signals.
    */
  private val selectedRuntimeIds = Var(Map.empty[String, String])

  private val createForm =
    RunConfigurationForm(
      toolArchitectures,
      service.allModels,
      loraService,
      browsers,
      assistantService.library.ofKind(PromptKind.AssistantSystem),
      runtimeService.runtimes
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
      runtimeService.selection.combineWith(loraService.loras)
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
            (selection, loras)
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
            id => sessionService.push(SessionService.Command.Stop(id))
          ).element
        }
    }

  private def sourceLabel(source: ModelSource): String = source match {
    case HuggingFace(repo, file)   => s"<hf:$repo/$file>"
    case ModelScope(repo, file)    => s"<modelscope:$repo/$file>"
    case Civitai(id, _, fileId, _) => s"<civitai:$id/$fileId>"
    case Local(path)               => path
  }

  private def handleCreate(): Unit = {
    val rm = createForm.snapshot()
    if (rm.id.nonEmpty && rm.label.nonEmpty && rm.architectureId.nonEmpty)
      service.push(Command.Create(rm))
  }

  private def handleDelete(id: String): Unit =
    service.push(Command.Delete(id))

  /** An untouched select means "the default": no explicit pick is sent, so the
    * backend resolves whatever the default runtime is at spawn time.
    */
  private def handleLaunch(configurationId: String): Unit =
    sessionService.push(
      SessionService.Command
        .Launch(configurationId, selectedRuntimeIds.now().get(configurationId))
    )

  private def startEdit(rm: RunConfiguration): Unit = {
    val architecture = archsNow.now().find(_.id == rm.architectureId)
    editCard.set(
      Some(
        RunConfigurationEditCard(
          rm,
          architecture,
          service.allModels,
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

  /** The launched session, if any \u2014 the page swaps to the generation UI
    * while one is live and back to the list when it stops. Keyed by session id,
    * so the panel (and its form state) survives the status socket's session
    * pushes and the `starting` \u2192 `ready` flip.
    */
  private val activeSessionKey: Signal[List[(String, String, RuntimeTool)]] =
    sessionService.sessions
      .map(
        _.values
          // Every live session this page covers, not just the first: with
          // both runners on one page an image model and an assistant are
          // routinely up together, and each wants its panel.
          .filter(s => s.status.isActive && tools.contains(s.tool))
          .toList
          .sortBy(_.startedAt)
          .map(session =>
            (session.id, session.runConfigurationId, session.tool)
          )
      )
      .distinct

  private def generationElement(
      sessionId: String,
      configurationId: String
  ): HtmlElement =
    GenerationPanel(
      sessionId = sessionId,
      configurationId = configurationId,
      configurationLabel = service.runConfigurations
        .map(
          _.find(_.id == configurationId)
            .map(_.label)
            .getOrElse(configurationId)
        )
        .distinct,
      sessionSignal =
        sessionService.sessions.map(_.get(configurationId)).distinct,
      architectureId = service.runConfigurations
        .map(_.find(_.id == configurationId).map(_.architectureId))
        .distinct,
      configurationLoras = service.runConfigurations
        .map(_.find(_.id == configurationId).map(_.loras).getOrElse(Nil))
        .distinct,
      configurationAssistantTemplateId = service.runConfigurations
        .map(_.find(_.id == configurationId).flatMap(_.assistantTemplateId))
        .distinct,
      targetArchitecture = service.runConfigurations
        .combineWith(service.architectures)
        .map { (configurations, architectures) =>
          configurations
            .find(_.id == configurationId)
            .flatMap(c => architectures.find(_.id == c.architectureId))
        }
        .distinct,
      service = generationService,
      assistantService = assistantService,
      loraService = loraService,
      projectService = projectService,
      logService = logService,
      onStop = () => sessionService.push(SessionService.Command.Stop(sessionId))
    ).element

  private def assistantElement(
      sessionId: String,
      configurationId: String
  ): HtmlElement =
    AssistantPanel(
      sessionId = sessionId,
      configurationLabel = service.runConfigurations
        .map(
          _.find(_.id == configurationId)
            .map(_.label)
            .getOrElse(configurationId)
        )
        .distinct,
      sessionSignal =
        sessionService.sessions.map(_.get(configurationId)).distinct,
      service = assistantService,
      generationService = generationService,
      onStop =
        () => sessionService.push(SessionService.Command.Stop(sessionId)),
      freePlay = true
    ).element

  private def listElement: HtmlElement = div(
    div(
      cls := "level",
      div(
        cls := "level-left",
        h1(cls := "title text-primary", "Run configurations")
      ),
      div(
        cls := "level-right",
        // Only while something is live, and only when the list is what is
        // being shown: the way back to a loaded model.
        child <-- activeSessionKey.map {
          case Nil      => emptyNode
          case sessions =>
            button(
              cls := "button is-link mr-2",
              s"↩ Back to ${sessions.size} running",
              onClick --> (_ => showList.set(false))
            )
        },
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
    child <-- showForm.signal.map {
      case false => emptyNode
      case true  =>
        configurationModal(
          "New run configuration",
          createForm.element,
          button(
            cls := "button is-success",
            span(cls := "plus-icon", "+"),
            " Create",
            onClick --> (_ => handleCreate())
          ),
          () => showForm.set(false)
        )
    },
    child <-- editCard.signal.map {
      case None       => emptyNode
      case Some(card) =>
        configurationModal(
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

  /** Creating or editing a configuration: the form in a modal, with the
    * service's errors inside it \u2014 a refused save would otherwise explain
    * itself behind the backdrop \u2014 and the confirming button beside Cancel.
    */
  private def configurationModal(
      title: String,
      form: HtmlElement,
      confirm: HtmlElement,
      onCancel: () => Unit
  ): HtmlElement =
    BrowserModal(
      title = Val(title),
      body = Seq(ErrorBanner(service), form),
      onCancel = onCancel,
      footerRight = div(
        cls := "buttons",
        confirm,
        button(cls := "button", "Cancel", onClick --> (_ => onCancel()))
      ),
      modalMods = Seq(
        documentEvents(_.onKeyDown).filter(_.key == "Escape")
          --> (_ => onCancel())
      ),
      cardMods = Seq(styleAttr := "width: min(60rem, 95vw);")
    ).element

  lazy val element: HtmlElement = div(
    cls := "content",
    service.effects,
    cacheService.effects,
    downloadService.effects,
    sessionService.effects,
    runtimeService.effects,
    prerequisites.followInstalls,
    generationService.effects,
    loraService.effects,
    downloadService.events --> Observer {
      case DownloadService.Event.Finished(_) =>
        cacheService.push(CacheService.Command.Load)
        downloadService.push(DownloadService.Command.Load)
    },
    service.architectures --> archsNow,
    service.events --> Observer {
      case Event.Created(_) => createForm.reset(); showForm.set(false)
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
    },
    ErrorBanner(service),
    ErrorBanner(service.modelService),
    ErrorBanner(service.architectureService),
    ErrorBanner(cacheService),
    ErrorBanner(downloadService),
    ErrorBanner(sessionService),
    ErrorBanner(runtimeService),
    ErrorBanner(generationService),
    ErrorBanner(loraService),
    // Live sessions take the page, as they always have - but the list stays
    // one click away rather than needing the session stopped to see it, and
    // the page's other tabs are reachable throughout.
    child <-- activeSessionKey.combineWith(showList.signal).map {
      case (Nil, _)          => listElement
      case (_, true)         => listElement
      case (sessions, false) =>
        div(
          cls := "live-panels",
          div(
            cls := "mb-3",
            button(
              cls := "button is-small",
              "← All configurations",
              onClick --> (_ => showList.set(true))
            )
          ),
          sessions.map {
            case (sessionId, configurationId, RuntimeTool.SdCpp) =>
              generationElement(sessionId, configurationId)
            case (sessionId, configurationId, RuntimeTool.LlamaCpp) =>
              assistantElement(sessionId, configurationId)
          }
        )
    }
  )
}
