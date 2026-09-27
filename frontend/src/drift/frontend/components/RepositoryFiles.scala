package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** One file of a repository, as the two repository browsers list them --
  * HuggingFace calls it `rfilename`, ModelScope `path`.
  */
case class BrowserFile(path: String, sizeBytes: Option[Long])

/** The Files tab of an opened repository, HuggingFace or ModelScope: weight
  * files are selectable, everything else is dimmed, and a split model is chosen
  * by its index, which brings its shards (`specs/34`).
  *
  * Installing LoRAs (`specs/33-lora-sources.md`), it behaves like the Civitai
  * version list: the repository is one group of files, ticked one by one --
  * none to begin with -- and the bar at the top installs them as one LoRA, as
  * one each, or into a LoRA already installed. Registering a model instead, a
  * row click picks that one file.
  */
class RepositoryFiles(
    files: Signal[List[BrowserFile]],
    loading: Signal[Boolean],
    onSelect: String => Unit,
    /** How the site gates this repository, when it does — the site's own word
      * for it ("auto", "manual" on HuggingFace).
      */
    gated: Signal[Option[String]] = Val(None),
    /** The site, for the notice: where the terms are to be accepted. */
    siteName: String = "the site",
    /** The chips one file carries, by path: the ✓ installed mark. */
    tagsOf: String => Seq[Mod[HtmlElement]] = _ => Nil,
    /** Install mode: the ticked files, and where they land. */
    onInstall: Option[(List[String], LoraGrouping) => Unit] = None,
    /** What they could join: the LoRAs this repository already made. */
    candidates: Signal[List[InstalledItem]] = Val(Nil)
) extends Component {

  private val installMode = onInstall.isDefined
  private val selection = FileSelection()

  private lazy val bar: Mod[HtmlElement] = selection.bar(candidates, onInstall)

  /** A gated repository lists its files like any other, and downloading one
    * answers 403 unless the token in Settings belongs to an account that has
    * been let in. Saying so here costs a line and saves a failed download.
    */
  private lazy val gateNotice: Signal[Option[HtmlElement]] = gated.map(
    _.map(kind =>
      div(
        cls := "notification is-warning is-light py-2 px-3 is-size-7 mb-2",
        span(cls := "has-text-weight-bold", "🔒 Gated repository. "),
        if (kind == "manual")
          s"$siteName lets each account in by hand: request access there, wait " +
            "for the author, then set a token in Settings."
        else
          s"Accept this repository's terms on $siteName and set a token in " +
            "Settings; downloads fail with 403 until then."
      )
    )
  )

  lazy val element: HtmlElement = div(
    bar,
    child.maybe <-- gateNotice,
    ResultsPlaceholder(
      busy = loading,
      isEmpty = files.map(_.isEmpty),
      busyText = "Loading files...",
      emptyText = "No model files found."
    ),
    children <-- files.combineWith(selection.files).map { (listed, ticked) =>
      val tickedKeys = ticked.map(_.key).toSet
      listed.map(file => row(file, tickedKeys))
    }
  )

  private def row(file: BrowserFile, ticked: Set[String]): HtmlElement =
    FileRow(
      name = file.path,
      onSelect = () =>
        if (installMode)
          selection.toggle(SelectedFile(file.path, file.sizeBytes))
        else onSelect(file.path),
      icon =
        if (installMode) selection.box(file.path, ticked) else FileRow.FileIcon,
      sizeBytes = file.sizeBytes,
      enabled = BrowserUtils.isModelFile(FileRow.extensionOf(file.path)) ||
        ShardedSafetensors.isIndex(file.path),
      selected = Val(installMode && ticked.contains(file.path)),
      tags = tagsOf(file.path)
    ).element
}
