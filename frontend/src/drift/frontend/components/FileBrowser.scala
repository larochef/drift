package drift.frontend.components

import drift.frontend.services.FileService
import drift.shared.*

import com.raquo.laminar.api.L.*

class FileBrowser(
    service: FileService,
    onSelect: String => Unit,
    onCancel: () => Unit,
    initialPath: String = "",
    /** Install mode (`specs/33-lora-sources.md`): the ticked files, and where
      * they land. A file here says nothing about where it belongs, so every
      * LoRA of the architecture is a candidate to join.
      */
    onInstall: Option[(List[String], LoraGrouping) => Unit] = None,
    candidates: Signal[List[InstalledItem]] = Val(Nil),
    /** The row of sources `SourceBrowser` puts in the modal's head. */
    sourceSwitch: Mod[HtmlElement] = emptyNode
) extends Component {
  private enum Filter { case Models; case All }

  private val installMode = onInstall.isDefined
  private val selection = FileSelection()

  private lazy val bar: Mod[HtmlElement] =
    selection.bar(candidates, onInstall, what = "files")

  val modelExts = Set(".safetensors", ".gguf", ".pt", ".pth", ".bin")

  private val navBus = new EventBus[String]
  private val currentPath = Var("")
  private val selectedPath = Var(Option.empty[String])
  private val filter = Var[Filter](Filter.Models)
  private val searchQuery = Var("")

  private def pathSegments(p: String): List[(String, String)] = {
    val cleaned = p.stripPrefix("/").stripSuffix("/")
    if (cleaned.isEmpty) return Nil
    val parts = cleaned.split("/").toList
    parts
      .scanLeft("") { (acc, seg) =>
        if (acc.isEmpty) s"/$seg" else s"$acc/$seg"
      }
      .drop(1)
      .zip(parts)
  }

  private val filteredEntries: Signal[List[FileEntry]] =
    Signal.combine(service.entries, filter.signal, searchQuery.signal).map {
      (es, f, q) =>
        val byType = f match {
          case Filter.All    => es
          case Filter.Models =>
            // A split model is registered by its index (`specs/34`).
            es.filter(e =>
              e.isDirectory || e.extension == ".gguf" || e.extension == ".safetensors" ||
                ShardedSafetensors.isIndex(e.name)
            )
        }
        if (q.isEmpty) byType
        else byType.filter(e => e.name.toLowerCase.contains(q.toLowerCase))
    }

  lazy val element: HtmlElement = BrowserModal(
    title = Val("Select a model file"),
    onCancel = onCancel,
    headerAction = div(
      cls := "browser-head-actions",
      sourceSwitch,
      button(cls := "delete", onClick --> (_ => onCancel()))
    ),
    modalMods = Seq(
      // The same chrome and the same width as the site browsers: this one is
      // a source among them (`specs/03`).
      cls := "browser-modal",
      service.effects,
      onMountCallback { _ =>
        if (initialPath.nonEmpty) navBus.writer.onNext(initialPath)
        else service.push(FileService.Command.LoadHome)
      },
      service.events --> Observer { case FileService.Event.HomeLoaded(path) =>
        navBus.writer.onNext(path)
      },
      navBus.events --> Observer[String] { path =>
        selectedPath.set(None)
        service.push(FileService.Command.ListDirectory(path))
      },
      navBus.events --> currentPath,
      currentPath.signal.changes --> Observer[String](_ => searchQuery.set(""))
    ),
    body = Seq(
      bar,
      div(
        cls := "field mb-2",
        child <-- currentPath.signal.map(renderBreadcrumb)
      ),
      // A live filter over the listing, not a search that is submitted, so
      // this one keeps its own input rather than using `SearchField`.
      div(
        cls := "field mb-2",
        div(
          cls := "control has-icons-left is-expanded",
          input(
            cls := "input is-small",
            typ := "text",
            placeholder := "Search files...",
            value <-- searchQuery,
            onInput.mapToValue --> searchQuery
          ),
          span(cls := "icon is-small is-left", "\uD83D\uDD0D")
        )
      ),
      hr(cls := "my-2"),
      ErrorBanner(service),
      children <-- filteredEntries
        .combineWith(currentPath.signal, selection.files)
        .map((entries, path, _) => renderEntries(entries, path))
    ),
    footerLeft = div(
      cls := "buttons",
      // Installing, the bar over the listing sends what is ticked; there is
      // nothing for a Select button to do.
      if (installMode) emptyNode
      else
        button(
          cls := "button is-success",
          "Select",
          disabled <-- selectedPath.signal.map(_.isEmpty),
          onClick --> { _ => selectedPath.now().foreach(onSelect) }
        ),
      BrowserModal.cancelButton(onCancel)
    ),
    footerRight = div(
      cls := "select is-small",
      select(
        defaultValue := "Models",
        onChange.mapToValue --> Observer[String] { v =>
          filter.set(if (v == "All") Filter.All else Filter.Models)
        },
        option(value := "Models", "Models (.gguf, .safetensors)"),
        option(value := "All", "All files")
      )
    )
  ).element

  private def renderBreadcrumb(path: String): HtmlElement = {
    val segments = pathSegments(path)
    div(
      cls := "breadcrumb has-slash-separator is-small mb-0",
      ul(
        li(
          a(
            cls := "text-primary",
            "/",
            onClick --> (_ => navBus.writer.onNext("/"))
          )
        ),
        segments.map { (fullPath, name) =>
          li(
            a(
              cls := "text-primary",
              name,
              onClick --> (_ => navBus.writer.onNext(fullPath))
            )
          )
        }
      )
    )
  }

  private def parentPath(p: String): String = {
    val normalized = p.stripSuffix("/")
    if (normalized.isEmpty || normalized == "/") "/"
    else {
      val idx = normalized.lastIndexOf('/')
      if (idx <= 0) "/" else normalized.substring(0, idx)
    }
  }

  private def renderEntries(
      es: List[FileEntry],
      path: String
  ): List[HtmlElement] = {
    val parentEntry =
      if (path == "/") Nil
      else {
        val parent = parentPath(path)
        List(
          FileRow(
            name = "..",
            onSelect = () => navBus.writer.onNext(parent),
            icon = FileRow.FolderIcon
          ).element
        )
      }
    parentEntry ++ es.map { entry =>
      FileRow(
        name = entry.name,
        onSelect = () =>
          if (entry.isDirectory) navBus.writer.onNext(entry.path)
          else if (installMode)
            selection.toggle(SelectedFile(entry.path, Some(entry.size)))
          else selectedPath.set(Some(entry.path)),
        icon =
          if (entry.isDirectory) FileRow.FolderIcon
          else if (!installMode) FileRow.FileIcon
          else selection.box(entry.path, selection.keys.toSet),
        sizeBytes = if (entry.isDirectory) None else Some(entry.size),
        selected =
          if (installMode) selection.isTicked(entry.path)
          else selectedPath.signal.map(_.contains(entry.path))
      ).element
    }
  }
}
