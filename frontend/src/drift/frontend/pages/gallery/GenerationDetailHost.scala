package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.frontend.services.*
import drift.frontend.services.GenerationService.Reuse
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** The full-size view of one generation, wired to the services once
  * (`specs/12-gallery.md`). The gallery opens it over its grid and the project
  * workspace over its results: the same modal, the same reuse offer, the same
  * upscale/resize rows and the same delete, so a result is a result wherever it
  * is clicked (François, 2026-09-09).
  *
  * `open` is the page's own state — deleting the shown generation clears it, so
  * a detail can never outlive its files — and `pool` is what the page has
  * loaded: the days on screen for the gallery, the project's generations for
  * the workspace. Lineage (the original this was made from, what was made from
  * it) is resolved inside that pool.
  *
  * The host mounts no `effects`: the pages already mount the services they
  * share, and a second mount would subscribe every stream twice.
  */
class GenerationDetailHost(
    open: Var[Option[GenerationDetailHost.Open]],
    pool: Signal[List[Generation]],
    historyService: HistoryService,
    runConfigurationService: RunConfigurationService,
    sessionService: SessionService,
    generationService: GenerationService,
    assistantService: AssistantService,
    postProcessService: PostProcessService,
    upscalerService: UpscalerService,
    /** The runtimes and the sd-cpp default: which runtime a PiD job gets, and
      * so how large its tiles are.
      */
    runtimeService: RuntimeService,
    /** What each configuration still has to download: its launch is offered as
      * that download until the weights are on disk (`specs/46`).
      */
    prerequisites: LaunchPrerequisites,
    /** The project a launch from here is made for, recorded on its session: the
      * workspace's; none from the gallery, whose launches land in the Sandbox
      * (`specs/47-sandbox.md`).
      */
    launchingProject: Option[String],
    /** What to do once a reuse is staged: the gallery leaves for the inference
      * page, the workspace fills the panel it already shows.
      */
    onReuseStaged: () => Unit,
    /** What to do once an image is staged for the assistant: the gallery leaves
      * for the assistant page, the workspace has it in the next column.
      */
    onAssistantStaged: () => Unit
) extends Component {

  /** Which section of a detail's right column is open, kept across openings —
    * the browsers keep their tab the same way (`specs/24`).
    */
  private val openSection = Var("parameters")

  /** Whether the right column is folded away — kept across openings like the
    * tab, since it follows what the pictures are shaped like rather than which
    * one is open.
    */
  private val panelHidden = Var(false)

  /** Which post-processing task the "Redraw & upscale" tab shows, kept across
    * openings too: a run of redraws over a batch is what the panel is opened
    * for, and picking the task again on every image would be in the way.
    */
  private val openTask = Var(PostProcessSection.RedrawTask)

  private val labels: Signal[Map[String, String]] =
    runConfigurationService.runConfigurations
      .map(_.map(rm => rm.id -> rm.label).toMap)
      .distinct

  /** An imported image, and whatever is made from it, has no configuration
    * (`specs/30`).
    */
  private def labelOf(labels: Map[String, String], id: String): String =
    if (id.isEmpty) "Imported" else labels.getOrElse(id, id)

  private val openGeneration: Signal[Option[GenerationDetailHost.Shown]] =
    open.signal.combineWith(pool).map { (opened, generations) =>
      opened.flatMap(o =>
        generations
          .find(_.id == o.generationId)
          .map(generation =>
            GenerationDetailHost.Shown(
              generation,
              o.outputIndex,
              RecordedParameters
                .sourcePromptOf(generation, id => generations.find(_.id == id))
            )
          )
      )
    }

  /** The configuration of the live generation session, if any — its id only,
    * `distinct`, so a session's progress ticks stop here instead of reaching
    * the offer.
    */
  private val liveGenerationConfiguration: Signal[Option[String]] =
    sessionService.sessions
      .map(
        _.values
          .filter(s => s.status.isActive && s.tool == RuntimeTool.SdCpp)
          .toList
          .sortBy(_.startedAt)
          .headOption
          .map(_.runConfigurationId)
      )
      .distinct

  /** Run configurations of generation architectures: only those can take a
    * task.
    */
  private val generationConfigurations: Signal[List[RunConfiguration]] =
    runConfigurationService.runConfigurations
      .combineWith(runConfigurationService.architectures)
      .map { (configurations, architectures) =>
        val generating = architectures
          .filter(_.tool == RuntimeTool.SdCpp)
          .map(_.id)
          .toSet
        configurations.filter(c => generating.contains(c.architectureId))
      }
      .distinct

  /** The diffusion upscalers: run configurations whose architecture is a pixel
    * diffusion decoder.
    */
  private val pidConfigurations: Signal[List[ConfigurationOption]] =
    configurationsOf(_.pixelDiffusionDecoder)

  /** What can redraw (`specs/27-redraw.md`): run configurations of sd-cpp
    * architectures tagged `image`.
    */
  private val redrawConfigurations: Signal[List[ConfigurationOption]] =
    configurationsOf(a =>
      a.tool == RuntimeTool.SdCpp && a.tags.contains("image")
    )

  /** What can edit by instruction (`specs/39-seamless-edit.md`): run
    * configurations of sd-cpp architectures tagged `edit`.
    */
  private val editConfigurations: Signal[List[ConfigurationOption]] =
    configurationsOf(a =>
      a.tool == RuntimeTool.SdCpp && a.tags.contains(ArchitectureTags.Edit)
    )

  private def configurationsOf(
      accepts: Architecture => Boolean
  ): Signal[List[ConfigurationOption]] =
    runConfigurationService.runConfigurations
      .combineWith(runConfigurationService.architectures)
      .map { (configurations, architectures) =>
        val accepted =
          architectures.filter(accepts).map(a => a.id -> a).toMap
        configurations
          .flatMap(rm =>
            accepted
              .get(rm.architectureId)
              .map(architecture =>
                ConfigurationOption(
                  rm.id,
                  rm.label,
                  referenceImages =
                    architecture.referenceImages == ReferenceImageUse.Context,
                  sizeMultiple = architecture.sizeMultiple,
                  runner = rm.runner
                )
              )
          )
          .sortBy(_.label)
      }
      .distinct

  /** Every live session, id → label, oldest first. */
  private val liveSessions: Signal[List[(String, String)]] =
    sessionService.sessions
      .combineWith(labels)
      .map { (sessions, labels) =>
        sessions.values
          .filter(_.status.isActive)
          .toList
          .sortBy(_.startedAt)
          .map(s => s.id -> labelOf(labels, s.runConfigurationId))
      }
      .distinct

  private def reuse(request: Reuse): Unit = {
    generationService.requestReuse(request)
    onReuseStaged()
  }

  /** The rule (`specs/12-gallery.md`): the same configuration live takes the
    * request field for field; any other configuration — live, or picked to be
    * launched — takes the *task*, so models can be compared on the same job.
    */
  private def offerFor(
      generation: Generation,
      /** The live generation session's configuration — an assistant session
        * cannot take a task.
        */
      live: Option[String],
      configurations: List[RunConfiguration],
      labels: Map[String, String],
      missing: Map[String, LaunchPrerequisites.Missing]
  ): ReuseOffer = {
    val ownLabel = labelOf(labels, generation.runConfigurationId)
    val others = configurations
      .filterNot(_.id == generation.runConfigurationId)
      .map(rm => rm.id -> rm.label)
      .sortBy(_._2)
    def launchOther(id: String, index: Int): Unit = {
      sessionService.push(
        SessionService.Command.Launch(id, None, launchingProject)
      )
      reuse(Reuse.Task(generation.ofOutput(index)))
    }
    live match {
      case Some(liveId) if liveId == generation.runConfigurationId =>
        ReuseOffer(
          List(
            ReuseAction(
              "↺ Reuse these parameters",
              "is-primary",
              index => reuse(Reuse.Full(generation.ofOutput(index)))
            )
          ),
          None
        )
      case Some(liveId) =>
        val liveLabel = labelOf(labels, liveId)
        ReuseOffer(
          List(
            ReuseAction(
              s"↺ Try this task on $liveLabel",
              "is-link",
              index => reuse(Reuse.Task(generation.ofOutput(index)))
            )
          ),
          Some(
            s"$liveLabel is running, not $ownLabel: the task and the " +
              "sampling settings shown carry over, its own defaults fill the " +
              "rest. Stop it to launch another configuration with this task."
          )
        )
      // What it lacks — a runtime to choose, weights to download — is offered
      // in the launch's place (`specs/46`).
      case None if missing.contains(generation.runConfigurationId) =>
        ReuseOffer(
          List.empty,
          Some(
            s"$ownLabel cannot launch yet: " +
              LaunchPrerequisites.describe(
                missing(generation.runConfigurationId)
              ) + "."
          ),
          others,
          launchOther,
          preparing = Some(generation.runConfigurationId)
        )
      case None
          if configurations.exists(_.id == generation.runConfigurationId) =>
        ReuseOffer(
          List(
            ReuseAction(
              s"▶ Launch $ownLabel and reuse these parameters",
              "is-primary",
              index => {
                sessionService.push(
                  SessionService.Command
                    .Launch(
                      generation.runConfigurationId,
                      None,
                      launchingProject
                    )
                )
                reuse(Reuse.Full(generation.ofOutput(index)))
              }
            )
          ),
          None,
          others,
          launchOther
        )
      case None =>
        ReuseOffer(
          List.empty,
          Some(
            s"Run configuration '${generation.runConfigurationId}' no longer " +
              "exists" +
              (if (others.isEmpty) "." else "; try its task on another model.")
          ),
          others,
          launchOther
        )
    }
  }

  lazy val element: HtmlElement = div(
    // The detail follows the listing: a deletion — from here or from a grid —
    // closes it, so a socket update can never show a stale copy.
    historyService.events --> Observer[HistoryService.Event] {
      case HistoryService.Event.Deleted(id) =>
        if (open.now().exists(_.generationId == id)) open.set(None)
      case HistoryService.Event.Imported(_) => ()
    },
    // Keyed on the generation and the output clicked, so a detail is built
    // once per opening and nothing else rebuilds it (bugs/24).
    children <-- openGeneration
      .map(_.toList)
      .split(shown => (shown.generation, shown.outputIndex))((_, shown, _) =>
        detailFor(shown)
      )
  )

  private def detailFor(shown: GenerationDetailHost.Shown): HtmlElement = {
    val generation = shown.generation
    GenerationDetail(
      generation,
      configurationLabel =
        labels.map(labelOf(_, generation.runConfigurationId)).distinct,
      // A derived entry offers no parameters to reuse — its original does —
      // and an imported image has none at all.
      offer =
        if (
          generation.derivation.isDefined ||
          generation.runConfigurationId.isEmpty
        ) Val(ReuseOffer(List.empty, None))
        else
          liveGenerationConfiguration
            .combineWith(
              generationConfigurations,
              labels,
              prerequisites.byConfiguration
            )
            .map((live, configurations, labels, missing) =>
              offerFor(generation, live, configurations, labels, missing)
            ),
      upscalers = upscalerService.upscalers.distinct,
      pidConfigurations = pidConfigurations,
      redrawConfigurations = redrawConfigurations,
      restorationTemplates =
        assistantService.library.ofKind(PromptKind.RedrawRestoration),
      editConfigurations = editConfigurations,
      editTemplates = assistantService.library.ofKind(PromptKind.Edit),
      liveSessions = liveSessions,
      jobs = postProcessService.visibleJobs,
      parent = pool
        .map(loaded =>
          generation.derivation.flatMap(d => loaded.find(_.id == d.parentId))
        )
        .distinct,
      derivatives = pool.distinct.map(
        _.filter(_.derivation.exists(_.parentId == generation.id))
          .sortBy(_.submittedAt)
      ),
      sourcePrompt = shown.sourcePrompt,
      onOpen = id => open.set(Some(GenerationDetailHost.Open(id))),
      onAskAssistant = (output, label) => {
        assistantService.attach(
          AssistantService.outputAttachment(generation, output, label)
        )
        onAssistantStaged()
      },
      onUpscale = (output, request) =>
        postProcessService.push(
          PostProcessService.Command
            .Upscale(output.date, output.fileName, request)
        ),
      onPid = (output, request) =>
        postProcessService.push(
          PostProcessService.Command
            .Pid(output.date, output.fileName, request)
        ),
      onRedraw = (output, request) =>
        postProcessService.push(
          PostProcessService.Command
            .Redraw(output.date, output.fileName, request)
        ),
      onEdit = (output, request) =>
        postProcessService.push(
          PostProcessService.Command
            .Edit(output.date, output.fileName, request)
        ),
      onCancelJob = jobId =>
        postProcessService.push(PostProcessService.Command.Cancel(jobId)),
      onPauseJob = (jobId, force) =>
        postProcessService.push(PostProcessService.Command.Pause(jobId, force)),
      onResumeJob = jobId =>
        postProcessService.push(PostProcessService.Command.Resume(jobId)),
      onDismissJob = jobId => postProcessService.dismiss(jobId),
      onStopSession =
        id => sessionService.push(SessionService.Command.Stop(id)),
      onClose = () => open.set(None),
      onDelete =
        () => GenerationDetailHost.confirmAndDelete(generation, historyService),
      openSection = openSection,
      openTask = openTask,
      panelHidden = panelHidden,
      prerequisites = prerequisites,
      initialOutputIndex = shown.outputIndex
    ).element
  }
}

object GenerationDetailHost {

  /** What the page has open: the generation, and which of its outputs the strip
    * starts on — a batch is shown tile by tile in the workspace, so the click
    * has to survive as far as the modal.
    */
  case class Open(generationId: String, outputIndex: Int = 0)

  /** The open detail read out of the URL tail (`<page>/<generation id>[/<output
    * index>]`).
    */
  def openOf(section: List[String]): Option[Open] =
    section.map(_.trim).filter(_.nonEmpty) match {
      case id :: index :: _ => Some(Open(id, index.toIntOption.getOrElse(0)))
      case id :: Nil        => Some(Open(id))
      case Nil              => None
    }

  /** That tail, for a page to append to its own path. */
  def tailOf(open: Option[Open]): String =
    open.fold("")(shown =>
      s"/${shown.generationId}" +
        (if (shown.outputIndex > 0) s"/${shown.outputIndex}" else "")
    )

  /** Keeps the open detail and the address bar in step, in both directions: a
    * refresh (or a link, or the back button) reopens what the URL names, and
    * opening or closing one writes it there. Which is the point — a repaint
    * takes minutes, and coming back to the page should come back to the image
    * (François, 2026-09-19).
    *
    * Each side checks before writing, so the two do not chase each other.
    */
  def boundToUrl(
      base: String,
      section: Signal[List[String]],
      open: Var[Option[Open]]
  ): Modifier[HtmlElement] =
    List(
      section --> Observer[List[String]] { parts =>
        val fromUrl = openOf(parts)
        if (fromUrl != open.now()) open.set(fromUrl)
      },
      open.signal.changes --> Observer[Option[Open]] { shown =>
        val path = base + tailOf(shown)
        if (window.location.pathname != path)
          frontroute.BrowserNavigation.pushState(url = path)
      }
    )

  /** An opening resolved in the loaded generations, with the prompt its PiD
    * panel starts from.
    */
  private case class Shown(
      generation: Generation,
      outputIndex: Int,
      sourcePrompt: String
  )

  /** How many files one generation owns on disk: its outputs, the input images
    * externalized beside them, and the sidecar.
    */
  def fileCount(generation: Generation): Int =
    generation.outputs.size +
      RecordedParameters.inputImages(generation).size + 1

  /** The one confirmation every delete goes through, wherever the button is: it
    * names the kind, the file count and the prompt, because the files go with
    * the entry and nothing here is undoable.
    */
  def confirmAndDelete(
      generation: Generation,
      historyService: HistoryService
  ): Unit = {
    val prompt = RecordedParameters.promptOf(generation).take(80)
    if (
      window.confirm(
        s"Delete this ${RecordedParameters.kindLabel(generation).toLowerCase} " +
          s"and its ${fileCount(generation)} file(s)?\n\n\"$prompt\""
      )
    )
      delete(generation, historyService)
  }

  /** Deletes without asking — for a caller that already confirmed a batch of
    * them. A generation with no output has no day to delete under.
    */
  def delete(generation: Generation, historyService: HistoryService): Unit =
    generation.outputs.headOption.foreach(output =>
      historyService.push(
        HistoryService.Command.Delete(output.date, generation.id)
      )
    )
}
