package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** The Files tab of an opened Civitai model: its versions, each with its base
  * model and what drift already has of it, and their weight files
  * (`specs/24-model-details-in-browsers.md`).
  *
  * Installing (`specs/33-lora-sources.md`) ticks files — none to begin with,
  * and across versions if that is what is wanted — and one bar at the top says
  * how many, how big, and whether they become one LoRA, one each, or files of a
  * LoRA already installed. Registering a model instead, a row click picks that
  * one file.
  */
class CivitaiVersionList(
    versions: Signal[List[CivitaiModelVersion]],
    cachedFileIds: Signal[Set[String]],
    loading: Signal[Boolean],
    civitaiBaseModels: List[String],
    installed: Signal[Installed],
    /** Install mode: the ticked files, and where they land. */
    onInstall: Option[(List[CivitaiFileRef], LoraGrouping) => Unit],
    /** Model mode: one file registers as one model. */
    onSelectFile: (CivitaiModelVersion, CivitaiModelFile) => Unit,
    /** What the ticked files could join: the LoRAs this model already made. */
    candidates: Signal[List[InstalledItem]] = Val(Nil)
) extends Component {

  private val installMode = onInstall.isDefined
  private val selection = FileSelection()

  /** A version published for another base model than the architecture's is
    * hidden: installing one is the commonest way to end up with a LoRA that
    * does nothing (François, 2026-09-18). The box brings them back — Civitai's
    * base model is whatever the author wrote, so it is a guide, not a rule.
    */
  private val everyVersion = Var(false)

  private def matchesArchitecture(version: CivitaiModelVersion): Boolean =
    civitaiBaseModels.isEmpty ||
      version.baseModel.forall(civitaiBaseModels.contains)

  /** Another model is opened: its files start unticked. */
  def clearSelection(): Unit = selection.clear()

  private def keyOf(version: CivitaiModelVersion, file: CivitaiModelFile) =
    s"${version.id}:${file.id}"

  private lazy val bar: Mod[HtmlElement] = selection.bar(
    candidates,
    onInstall.map(install =>
      (keys, grouping) =>
        install(
          keys.flatMap(_.split(":", 2) match {
            case Array(versionId, fileId) =>
              Some(CivitaiFileRef(versionId, fileId))
            case _ => None
          }),
          grouping
        )
    )
  )

  lazy val element: HtmlElement = div(
    bar,
    ResultsPlaceholder(
      busy = loading,
      isEmpty = versions.map(_.isEmpty),
      busyText = "Loading files...",
      emptyText = "No model files found."
    ),
    child.maybe <-- versions
      .combineWith(everyVersion.signal)
      .map(otherBaseModels),
    children <-- versions
      .combineWith(cachedFileIds, everyVersion.signal, selection.files)
      .map { (shown, cachedIds, every, ticked) =>
        val listed = if (every) shown else shown.filter(matchesArchitecture)
        val tickedKeys = ticked.map(_.key).toSet
        listed.flatMap { version =>
          versionHeader(version) +:
            version.files
              .map(file => fileRow(version, file, cachedIds, tickedKeys))
        }
      }
  )

  /** What the architecture's base models leave out, and the box that shows it
    * anyway.
    */
  private def otherBaseModels(
      shown: List[CivitaiModelVersion],
      every: Boolean
  ): Option[HtmlElement] = {
    val hidden = shown.count(version => !matchesArchitecture(version))
    Option.when(civitaiBaseModels.nonEmpty && (hidden > 0 || every))(
      div(
        cls := "mb-2",
        BrowserFilters
          .checkbox(
            s"Versions for other base models ($hidden)",
            s"This model's other versions were published for base models this " +
              s"architecture does not declare (it declares " +
              s"${civitaiBaseModels.mkString(", ")}). They install all the " +
              "same, and usually do nothing.",
            everyVersion
          )
          .amend(cls := "ml-0"),
        Option.when(hidden == shown.size && !every)(
          p(
            cls := "text-secondary is-size-7 mt-1",
            s"This model publishes no version for " +
              s"${civitaiBaseModels.mkString(", ")}."
          )
        )
      )
    )
  }

  private def versionHeader(version: CivitaiModelVersion): HtmlElement =
    div(
      cls := "version-section",
      div(
        cls := "level is-mobile mb-2",
        div(
          cls := "level-left",
          h3(
            cls := "is-size-6 has-text-weight-bold has-text-primary",
            version.name
          ),
          CivitaiTags.baseModel(version, civitaiBaseModels),
          CivitaiTags.paidAccess(version),
          Installed.mark(
            installed,
            _.fromCivitaiVersion(version.id.toString),
            cls := "ml-2"
          )
        )
      )
    )

  private def fileRow(
      version: CivitaiModelVersion,
      file: CivitaiModelFile,
      cachedIds: Set[String],
      ticked: Set[String]
  ): HtmlElement = {
    val key = keyOf(version, file)
    val sizeBytes = file.sizeKB.map(kb => (kb * 1024L).toLong)
    FileRow(
      name = file.name,
      onSelect = () =>
        if (installMode) selection.toggle(SelectedFile(key, sizeBytes))
        else onSelectFile(version, file),
      // Installing, a row click ticks the file for the bar; registering a
      // model, it picks that one file.
      icon = if (installMode) selection.box(key, ticked) else FileRow.FileIcon,
      sizeBytes = sizeBytes,
      enabled = BrowserUtils.isModelFile(FileRow.extensionOf(file.name)),
      selected = Val(installMode && ticked.contains(key)),
      tags = CivitaiTags.ofFile(file, cachedIds.contains(file.id.toString)) :+
        Installed.mark(
          installed,
          _.fromCivitaiFile(file.id.toString),
          cls := "ml-1"
        )
    ).element
  }
}
