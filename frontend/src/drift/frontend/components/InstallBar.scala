package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** What the ticked files add up to, where they will land, and the button that
  * sends them (`specs/33-lora-sources.md`) — one button for the whole opened
  * model or repository, whatever the site.
  *
  * Several files travel together by default, since that is what makes a wan 2.2
  * pair one LoRA — but files of several releases (Civitai versions), none of
  * them a high- or low-noise half, are alternatives, and install as separate
  * LoRAs by default: applying a LoRA applies every file it holds. One file
  * makes a LoRA of its own; and either can join a LoRA already installed, which
  * is how a pair fetched one half at a time ends up whole.
  */
class InstallBar(
    selection: FileSelection,
    /** The LoRAs these files could join: the ones already holding a file from
      * this model or repository.
      */
    candidates: Signal[List[InstalledItem]],
    onInstall: LoraGrouping => Unit,
    /** What the files are called here, for the empty line. */
    what: String = "files"
) extends Component {

  /** "together", "separate", or the id of the LoRA to join, once picked. */
  private val target = Var(Option.empty[String])

  lazy val element: HtmlElement = div(
    cls := "install-bar",
    child <-- selection.files
      .combineWith(candidates, selection.sent)
      .map(bar)
  )

  private def bar(
      files: List[SelectedFile],
      choices: List[InstalledItem],
      sent: Option[Int]
  ): HtmlElement =
    if (files.isEmpty)
      p(
        cls := "is-size-7 text-secondary",
        sent match {
          case Some(count) =>
            s"✓ $count ${
                if (count == 1) "file" else "files"
              } sent to install " +
              "— the download shows on the architecture's card."
          case None => s"Tick the $what to install, from any version."
        }
      )
    else {
      val bytes = files.flatMap(_.sizeBytes).sum
      val size =
        if (bytes > 0) s" (${BrowserUtils.formatSize(bytes)})" else ""
      div(
        cls := "is-flex is-align-items-center",
        styleAttr := "gap: 0.75rem; flex-wrap: wrap;",
        BrowserFilters.choice(
          options(files, choices),
          chosen(files, choices),
          "install as",
          picked => target.set(Some(picked))
        ),
        button(
          cls := "button is-primary is-small",
          s"⬇ Install ${files.size} ${
              if (files.size == 1) "file" else "files"
            }$size",
          onClick --> { _ =>
            onInstall(grouping(chosen(files, choices)))
            selection.sendOff()
          }
        )
      )
    }

  private def options(
      files: List[SelectedFile],
      choices: List[InstalledItem]
  ): List[(String, String)] =
    List(
      "together" ->
        (if (files.sizeIs == 1) "a new LoRA"
         else s"one LoRA of ${files.size} files")
    ) ++
      Option
        .when(files.sizeIs > 1)("separate" -> s"${files.size} separate LoRAs")
        .toList ++
      choices.map(item => item.id -> s"files of '${item.name}'")

  /** The chosen target, or the default when none was or what was chosen is gone
    * — another repository was opened, or the selection came down to one file.
    */
  private def chosen(
      files: List[SelectedFile],
      choices: List[InstalledItem]
  ): String = {
    val offered = options(files, choices).map((key, _) => key).toSet
    target.now().filter(offered.contains).getOrElse(byDefault(files))
  }

  /** Separate for files of several releases none of which is a stage's half,
    * together otherwise.
    */
  private def byDefault(files: List[SelectedFile]): String =
    if (
      files.flatMap(_.release).distinct.sizeIs > 1 &&
      files.forall(_.stage == LoraFileStage.General)
    ) "separate"
    else "together"

  private def grouping(value: String): LoraGrouping = value match {
    case "together" => LoraGrouping.Together
    case "separate" => LoraGrouping.Separate
    case loraId     => LoraGrouping.Into(loraId)
  }
}
