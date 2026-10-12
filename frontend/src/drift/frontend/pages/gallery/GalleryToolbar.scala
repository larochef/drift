package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** The gallery's toolbar: the filters' fields, the import button and the switch
  * into selecting.
  */
final class GalleryToolbar(
    filters: GalleryFilters,
    projects: Signal[List[Project]],
    selection: GallerySelection,
    importer: GalleryImport
) extends Component {
  import filters.{
    configurationFilter,
    configurationOptions,
    kindFilter,
    nsfwProjectIds,
    projectFilter,
    searchVar,
    showNsfw
  }

  lazy val element: HtmlElement = div(
    cls := "gallery-toolbar mb-4",
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Project"),
      div(
        cls := "select is-small",
        select(
          onChange.mapToValue --> projectFilter,
          option(value := "", "All projects"),
          children <-- projects
            .combineWith(showNsfw.signal)
            .map((projects, nsfwShown) =>
              projects
                .filter(project => nsfwShown || !project.nsfw)
                .map(project =>
                  option(
                    value := project.id,
                    selected <-- projectFilter.signal.map(_ == project.id),
                    project.label
                  )
                )
            )
        )
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "NSFW"),
      label(
        cls := "checkbox text-secondary is-size-7",
        input(
          typ := "checkbox",
          checked <-- showNsfw.signal,
          onChange.mapToChecked.compose(
            _.withCurrentValueOf(nsfwProjectIds)
          ) --> Observer[(Boolean, Set[String])] { case (shown, nsfwProjects) =>
            showNsfw.set(shown)
            // A project that is hidden again cannot stay the one filtered on.
            if (!shown && nsfwProjects(projectFilter.now()))
              projectFilter.set("")
          }
        ),
        " Show NSFW projects"
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Run configuration"),
      div(
        cls := "select is-small",
        select(
          onChange.mapToValue --> configurationFilter,
          option(value := "", "All configurations"),
          children <-- configurationOptions.map(_.map { (id, label) =>
            option(
              value := id,
              selected <-- configurationFilter.signal.map(_ == id),
              label
            )
          })
        )
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Kind"),
      div(
        cls := "select is-small",
        select(
          onChange.mapToValue --> kindFilter,
          option(value := "", "Images and videos"),
          option(value := "img_gen", "Images"),
          option(value := "vid_gen", "Videos")
        )
      )
    ),
    div(
      cls := "field",
      styleAttr := "flex: 1 1 16rem;",
      label(cls := "label text-primary is-small", "Search prompts"),
      input(
        cls := "input is-small",
        typ := "search",
        placeholder := "words from the prompt",
        value <-- searchVar.signal,
        onInput.mapToValue --> searchVar
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Import"),
      label(
        cls := "button is-small",
        title := "PNG or JPEG images and WebM, MP4, MOV or Matroska videos " +
          "from your computer, to upscale, redraw, edit or use as inputs — " +
          "or drop them on the gallery",
        input(
          typ := "file",
          accept := "image/png,image/jpeg,video/webm,video/mp4," +
            "video/quicktime,video/x-matroska,.mkv",
          multiple := true,
          styleAttr := "display: none;",
          onChange --> { event =>
            val field = event.target.asInstanceOf[dom.HTMLInputElement]
            importer.importFiles(field.files)
            // The same file picked again must fire again.
            field.value = ""
          }
        ),
        "⤓ Import images or videos"
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Cleanup"),
      button(
        cls <-- selection.selecting.signal.map(on =>
          if (on) "button is-small is-primary" else "button is-small"
        ),
        child.text <-- selection.selecting.signal.map(on =>
          if (on) "Done selecting" else "☑ Select"
        ),
        onClick --> (_ => selection.toggleSelecting())
      )
    )
  )
}
