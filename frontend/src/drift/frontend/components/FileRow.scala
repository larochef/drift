package drift.frontend.components

import com.raquo.laminar.api.L.*

/** One `file-entry` row: icon, name, optional size. Used by all three browsers
  * -- the HuggingFace and Civitai file lists and the local directory listing --
  * which previously each carried their own copy of this markup.
  */
class FileRow(
    name: String,
    onSelect: () => Unit,
    icon: String = FileRow.FileIcon,
    sizeBytes: Option[Long] = None,
    enabled: Boolean = true,
    selected: Signal[Boolean] = Val(false),
    /** Chips rendered between the name and the size — the Civitai browser puts
      * precision/quant markers here so two files with the same name can be told
      * apart, and the repository browsers the ✓ installed mark.
      */
    tags: Seq[Mod[HtmlElement]] = Seq.empty
) extends Component {
  lazy val element: HtmlElement =
    div(
      cls := "file-entry",
      cls("is-dimmed") <-- Val(!enabled),
      cls("is-selected") <-- selected,
      onClick --> (_ => if (enabled) onSelect()),
      span(cls := "file-entry-icon", icon),
      span(cls := "file-entry-name text-primary", name),
      tags,
      sizeBytes match {
        case Some(bytes) =>
          span(
            cls := "file-entry-size text-secondary is-size-7",
            BrowserUtils.formatSize(bytes)
          )
        case None => emptyNode
      }
    )
}

object FileRow {
  val FolderIcon = "📁"
  val FileIcon = "📄"

  /** The extension of `name`, lowercased and including the dot, or "". */
  def extensionOf(name: String): String =
    name.lastIndexOf('.') match {
      case i if i >= 0 => name.substring(i).toLowerCase
      case _           => ""
    }
}
