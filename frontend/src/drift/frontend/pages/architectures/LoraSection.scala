package drift.frontend.pages.architectures

import drift.frontend.components.*
import drift.frontend.services.{BrowserServices, LoraService}
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** The LoRA collection of one architecture (`specs/09-lora-management.md`):
  * install from Civitai, HuggingFace or the disk, or one of the official LoRAs
  * drift offers (`specs/33-lora-sources.md`), and edit the things worth
  * remembering — default strength, the sfw/nsfw placement, a wrongly guessed
  * wan stage.
  */
class LoraSection(
    browsers: BrowserServices,
    architecture: Architecture,
    service: LoraService
) extends Component {

  /** Folded by default, so the architecture cards keep a similar size however
    * many LoRAs each holds.
    */
  private val open = Var(false)

  /** The LoRA whose pairing panel is showing, the half it would join, and what
    * the two would be called together.
    */
  private val pairing = Var(Option.empty[String])

  /** The LoRA whose name is being rewritten. A file's name is what an install
    * has to go on, and `high_noise_model` says nothing a week later (François,
    * 2026-09-18).
    */
  private val renaming = Var(Option.empty[String])

  /** The LoRAs whose sampling block is open: kept here, since a row is drawn
    * again each time its LoRA is saved.
    */
  private val samplingOpen = Var(Set.empty[String])

  /** Two experts (wan 2.2): a LoRA then sets the high-noise expert's too. */
  private val twoExperts = architecture.checkpoints.exists(
    _.flag == "--high-noise-diffusion-model"
  )
  private val partner = Var("")
  private val pairName = Var("")

  /** A chat model's LoRA has no diffusion stage to cycle: its chip names the
    * file instead (`specs/35-assistant-loras.md`).
    */
  private val chatModel = architecture.tool == RuntimeTool.LlamaCpp

  private val architectureLoras: Signal[List[Lora]] =
    service.loras.map(_.filter(_.architectureId == architecture.id))

  /** The official LoRAs for this architecture that are not installed. */
  private val offered: Signal[List[Lora]] =
    service.catalog.combineWith(architectureLoras).map { (catalog, installed) =>
      val installedIds = installed.map(_.id).toSet
      catalog.filter(entry =>
        entry.architectureId == architecture.id &&
          !installedIds.contains(entry.id)
      )
    }

  private def stageLabel(stage: LoraFileStage): String = stage match {
    case LoraFileStage.General   => "general"
    case LoraFileStage.LowNoise  => "low noise"
    case LoraFileStage.HighNoise => "high noise"
  }

  private def nextStage(stage: LoraFileStage): LoraFileStage = stage match {
    case LoraFileStage.General   => LoraFileStage.LowNoise
    case LoraFileStage.LowNoise  => LoraFileStage.HighNoise
    case LoraFileStage.HighNoise => LoraFileStage.General
  }

  private def cycleStage(lora: Lora, file: LoraFile): Unit =
    service.push(
      LoraService.Command.Update(
        lora.copy(files =
          lora.files.map(f =>
            if (f.fileName == file.fileName) f.copy(stage = nextStage(f.stage))
            else f
          )
        )
      )
    )

  /** One half of a wan 2.2 pair: every file of this LoRA is high noise, or
    * every file is low noise. A LoRA holding both, or holding general files, is
    * not a half of anything.
    */
  private def loneStage(lora: Lora): Option[LoraFileStage] =
    lora.files.map(_.stage).distinct match {
      case List(stage) if stage != LoraFileStage.General => Some(stage)
      case _                                             => None
    }

  /** The other halves it could join: this architecture's LoRAs that are all of
    * the opposite stage (François, 2026-09-18: ModelScope publishes the two
    * halves of a pair as two repositories, so they install as two LoRAs).
    */
  private def partnersOf(lora: Lora, all: List[Lora]): List[Lora] =
    loneStage(lora).toList.flatMap { stage =>
      val wanted =
        if (stage == LoraFileStage.HighNoise) LoraFileStage.LowNoise
        else LoraFileStage.HighNoise
      all.filter(other =>
        other.id != lora.id && loneStage(other).contains(wanted)
      )
    }

  /** What the pair could be called: what the two names have in common, which is
    * usually the name without the half that differs.
    */
  private def suggestedName(one: Lora, other: Lora): String = {
    val common = one.label
      .zip(other.label)
      .takeWhile((left, right) => left == right)
      .map((left, _) => left)
      .mkString
      .trim
      .replaceAll("[\\s\\-_:·(]+$", "")
      .trim
    if (common.length >= 4) common else one.label
  }

  private def startPairing(lora: Lora, partners: List[Lora]): Unit =
    partners.headOption.foreach { first =>
      partner.set(first.id)
      pairName.set(suggestedName(lora, first))
      pairing.set(Some(lora.id))
    }

  /** The name follows the chosen half, unless the user has written their own.
    */
  private def choosePartner(
      lora: Lora,
      partners: List[Lora],
      picked: String
  ): Unit = {
    val was = partners.find(_.id == partner.now())
    partner.set(picked)
    partners
      .find(_.id == picked)
      .foreach(chosen =>
        if (was.forall(one => pairName.now() == suggestedName(lora, one)))
          pairName.set(suggestedName(lora, chosen))
      )
  }

  private def pairPanel(lora: Lora, partners: List[Lora]): HtmlElement =
    div(
      cls := "mt-2",
      p(
        cls := "is-size-7 text-secondary mb-1",
        "Both files land in this LoRA, each keeping its stage; the other " +
          "entity goes."
      ),
      div(
        cls := "is-flex is-align-items-center",
        styleAttr := "gap: 0.5rem; flex-wrap: wrap;",
        BrowserFilters.choice(
          partners.map(one => one.id -> one.label),
          partner.now(),
          "the other half",
          picked => choosePartner(lora, partners, picked)
        ),
        input(
          cls := "input is-small",
          styleAttr := "max-width: 18rem;",
          placeholder := "Name for the pair",
          value <-- pairName,
          onInput.mapToValue --> pairName
        ),
        button(
          cls := "button is-small is-success",
          "Pair",
          disabled <-- pairName.signal.map(_.trim.isEmpty),
          onClick --> { _ =>
            service.push(
              LoraService.Command
                .Pair(lora.id, partner.now(), pairName.now().trim)
            )
            pairing.set(None)
          }
        ),
        button(
          cls := "button is-small",
          "Cancel",
          onClick --> (_ => pairing.set(None))
        )
      )
    )

  /** The name, while it is being rewritten: Enter or leaving it saves, Escape
    * puts it back.
    */
  private def renameField(lora: Lora): HtmlElement =
    input(
      cls := "input is-small",
      styleAttr := "max-width: 22rem;",
      defaultValue := lora.label,
      onMountFocus,
      inContext { field =>
        Seq(
          onKeyDown.filter(_.key == "Enter") --> (_ =>
            rename(lora, field.ref.value)
          ),
          onKeyDown.filter(_.key == "Escape") --> (_ => renaming.set(None)),
          onBlur --> (_ => rename(lora, field.ref.value))
        )
      }
    )

  /** Saves the new name, once: Enter takes the field away, and the blur that
    * follows must not send it again.
    */
  private def rename(lora: Lora, value: String): Unit =
    if (renaming.now().contains(lora.id)) {
      val name = value.trim
      renaming.set(None)
      if (name.nonEmpty && name != lora.label)
        service.push(LoraService.Command.Update(lora.copy(label = name)))
    }

  private def loraRow(
      lora: Lora,
      jobs: List[LoraDownloadJob],
      all: List[Lora]
  ): HtmlElement = {
    val partners = partnersOf(lora, all)
    val activeJobs = jobs.filter(j => j.loraId == lora.id)
    div(
      cls := "box bg-table-header py-2 px-3 mb-2",
      div(
        cls := "level is-mobile mb-1",
        div(
          cls := "level-left",
          div(
            p(
              cls := "text-primary has-text-weight-bold",
              ProviderIcon.of(lora).map(_.amend(cls := "mr-2")),
              child <-- renaming.signal.map {
                case Some(id) if id == lora.id => renameField(lora)
                case _                         =>
                  span(
                    cls := "cursor-pointer",
                    title := "Click to rename — the file's name is what an " +
                      "install had to go on",
                    lora.label,
                    onClick --> (_ => renaming.set(Some(lora.id)))
                  )
              }
            ),
            if (lora.triggerWords.nonEmpty)
              p(
                cls := "is-size-7 text-secondary",
                s"triggers: ${lora.triggerWords.mkString(", ")}"
              )
            else emptyNode,
            if (lora.sampling.nonEmpty)
              p(
                cls := "is-size-7 text-secondary",
                s"sets: ${lora.sampling.summary.mkString(" · ")}"
              )
            else emptyNode
          )
        ),
        div(
          cls := "level-right",
          span(
            cls := s"tag is-small mr-2 ${
                if (lora.nsfw) "is-danger" else "is-success"
              }",
            styleAttr := "cursor: pointer;",
            title := "Click to move between the sfw and nsfw folders — a " +
              "running session keeps seeing the old path until restart",
            if (lora.nsfw) "nsfw" else "sfw",
            onClick --> (_ =>
              service.push(
                LoraService.Command.Update(lora.copy(nsfw = !lora.nsfw))
              )
            )
          ),
          label(
            cls := "is-size-7 text-secondary mr-1",
            "strength"
          ),
          input(
            cls := "input is-small mr-2",
            styleAttr := "width: 4.5rem;",
            typ := "number",
            stepAttr := "0.05",
            defaultValue := lora.defaultStrength.toString,
            title := "Default strength — remembered, some LoRAs need " +
              "specific values",
            onChange.mapToValue --> { value =>
              value.toDoubleOption.foreach(strength =>
                service.push(
                  LoraService.Command
                    .Update(lora.copy(defaultStrength = strength))
                )
              )
            }
          ),
          Option.when(partners.nonEmpty)(
            button(
              cls := "button is-small mr-2",
              "⇄ Pair",
              title := "This LoRA holds one stage only: join it with the " +
                "other half, published on its own",
              onClick --> (_ => startPairing(lora, partners))
            )
          ),
          Option.unless(chatModel)(
            button(
              cls := "button is-small mr-2",
              "⚙ Sampling",
              title := "The steps, CFG and flow shift this LoRA was made " +
                "for: selecting it sets them in the generation form",
              onClick --> (_ =>
                samplingOpen.update(open =>
                  if (open.contains(lora.id)) open - lora.id
                  else open + lora.id
                )
              )
            )
          ),
          button(
            cls := "button is-danger is-small",
            "🗑️",
            title := "Delete the LoRA and its files",
            onClick --> (_ =>
              if (window.confirm(s"Delete LoRA '${lora.label}' and its files?"))
                service.push(LoraService.Command.Delete(lora.id))
            )
          )
        )
      ),
      div(
        cls := "tags mb-0",
        lora.files.map { file =>
          val job = activeJobs.find(_.fileName == file.fileName)
          span(
            cls := "tag is-small is-info",
            if (chatModel) Seq(title := s"${file.displayName}, ${file.origin}")
            else
              Seq(
                styleAttr := "cursor: pointer;",
                title := s"${file.displayName}, ${file.origin} — click to " +
                  "change the stage (wan 2.2 pairs are low + high noise)"
              ),
            if (chatModel) file.displayName else stageLabel(file.stage),
            job.map(j =>
              j.state match {
                case DownloadState.Downloading =>
                  val percent = j.totalBytes
                    .filter(_ > 0)
                    .map(total =>
                      s" ${(j.downloadedBytes * 100 / total).min(100)}%"
                    )
                    .getOrElse("…")
                  span(cls := "ml-1", s"⬇$percent")
                case DownloadState.Queued => span(cls := "ml-1", "⬇ queued")
                case DownloadState.Failed => span(cls := "ml-1", "✗ failed")
                case _                    => emptyNode
              }
            ),
            onClick --> (_ => if (!chatModel) cycleStage(lora, file))
          )
        }
      ),
      activeJobs
        .flatMap(_.error)
        .headOption
        .map(reason => p(cls := "is-size-7 has-text-danger", reason)),
      child <-- pairing.signal.map {
        case Some(id) if id == lora.id => pairPanel(lora, partners)
        case _                         => emptyNode
      },
      child <-- samplingOpen.signal.map(_.contains(lora.id)).distinct.map {
        case true =>
          new LoraSamplingFields(lora, twoExperts, service).element
        case false => emptyNode
      }
    )
  }

  private def offeredRow(entry: Lora): HtmlElement = {
    val size = entry.files.flatMap(_.sizeBytes).sum
    div(
      cls := "box bg-table-header py-2 px-3 mb-2",
      div(
        cls := "is-flex is-align-items-center",
        styleAttr := "gap: 0.5rem;",
        div(
          styleAttr := "flex: 1 1 auto; min-width: 0; overflow-wrap: anywhere;",
          p(
            cls := "text-primary has-text-weight-bold",
            ProviderIcon.of(entry).map(_.amend(cls := "mr-2")),
            entry.label
          ),
          entry.description.map(text =>
            p(cls := "is-size-7 text-secondary", text)
          ),
          p(
            cls := "is-size-7 text-secondary",
            entry.files.map(_.origin).mkString(", ")
          )
        ),
        button(
          cls := "button is-small is-info",
          styleAttr := "flex: 0 0 auto;",
          "Install",
          title :=
            (if (size > 0) s"Download ${BrowserUtils.formatSize(size)}"
             else "Download"),
          onClick --> (_ =>
            service.push(
              LoraService.Command.Install(
                architecture.id,
                LoraInstallSource.Catalog(entry.id)
              )
            )
          )
        )
      )
    )
  }

  lazy val element: HtmlElement = FoldedSection(
    title = architectureLoras.map(loras => s"LoRAs (${loras.size})"),
    open = open,
    // A download in progress stays visible while the section is folded.
    note = child <-- architectureLoras.combineWith(service.jobs).map {
      (loras, jobs) =>
        val ids = loras.map(_.id).toSet
        val downloading = jobs.count(job =>
          ids.contains(job.loraId) &&
            (job.state == DownloadState.Downloading ||
              job.state == DownloadState.Queued)
        )
        if (downloading == 0) emptyNode
        else
          span(
            cls := "text-secondary is-size-7 ml-2",
            s"⬇ $downloading downloading"
          )
    },
    // Opening the browser opens the section too: install progress shows on the
    // rows.
    action = LoraInstallButton(
      browsers,
      architecture,
      service,
      onOpen = () => open.set(true)
    ).element,
    body = div(
      children <-- architectureLoras
        .combineWith(service.jobs, offered)
        .map { (loras, jobs, offers) =>
          val installed =
            if (loras.isEmpty)
              List(
                p(
                  cls := "text-secondary is-size-7",
                  "No LoRAs installed for this architecture yet."
                )
              )
            else loras.map(lora => loraRow(lora, jobs, loras))
          val official =
            if (offers.isEmpty) Nil
            else
              p(
                cls := "label text-primary is-size-7 mt-3 mb-1",
                "Official LoRAs"
              ) :: offers.map(offeredRow)
          installed ++ official
        }
    )
  ).element.amend(cls := "field mt-3")
}
