package drift.frontend.components

import drift.frontend.services.CivitaiService
import drift.shared.*

import com.raquo.airstream.state.StrictSignal
import com.raquo.laminar.api.L.*

/** The Images tab of an opened Civitai model (`specs/24`), one version at a
  * time: first the images its author published with the version — already in
  * the payload, so they show at once — then posts from Civitai's image search,
  * ten at a time, since that search is slow, often overloaded and ranks
  * somewhat at random (François, 2026-09-11). A tile opens a viewer linking to
  * its page on civitai.com; the API returns posts without their recipes.
  *
  * Rebuilt each time the tab is shown; the service keeps the posts it last
  * loaded, so coming back to the tab for the same model reloads nothing.
  */
class CivitaiImageGallery(
    service: CivitaiService,
    modelId: Int,
    versions: StrictSignal[List[CivitaiModelVersion]],
    /** The browser's "Include NSFW" switch: the gallery follows it. */
    includeNsfw: Var[Boolean]
) extends Component {
  import CivitaiImageGallery.{GalleryItem, Sorts}

  // The last query for this model, so the filters come back as they were;
  // otherwise the newest version. Always a version, as on civitai.com's model
  // page: the per-model query took 7-20 s when probed (2026-09-11) — past the
  // server's request timeout — and answered nothing for a LoRA whose versions
  // all have posts, while a per-version one answers in about a second.
  private val previous = service.currentImageQuery.filter(_.modelId == modelId)
  private val versionFilter = Var(
    previous
      .map(_.versionId)
      .getOrElse(versions.now().find(_.id != 0).map(_.id))
  )
  private val sort = Var(previous.map(_.sort).getOrElse(Sorts.head))
  private val viewing = Var(Option.empty[GalleryItem])

  /** The version's own images, filtered as the search is: level 1 is Civitai's
    * PG, the only one left without NSFW; `minor`-flagged ones are skipped, as
    * the search cards' previews already skip them.
    */
  private val showcase: Signal[List[GalleryItem]] =
    versions.combineWith(versionFilter.signal, includeNsfw.signal).map {
      (all, versionId, nsfw) =>
        all
          .find(version => versionId.contains(version.id))
          .toList
          .flatMap(version =>
            version.images
              .filter(image => !image.minor && (nsfw || image.nsfwLevel == 1))
              .map(image =>
                GalleryItem(
                  url = image.url,
                  video = image.`type` == "video",
                  username = None,
                  page = image.id.fold(
                    s"https://civitai.com/models/$modelId?modelVersionId=${version.id}"
                  )(id => s"https://civitai.com/images/$id")
                )
              )
          )
    }

  /** The version's images first, then the posts not already among them — the
    * search returns the author's own posts too.
    */
  private val items: Signal[List[GalleryItem]] =
    showcase.combineWith(service.images).map { (own, posts) =>
      val seen = own.map(_.key).toSet
      own ++ posts
        .map(post =>
          GalleryItem(
            url = post.url,
            video = post.isVideo,
            username = post.username,
            page = s"https://civitai.com/images/${post.id}"
          )
        )
        .filterNot(item => seen.contains(item.key))
    }

  private def load(): Unit = {
    val query = CivitaiService.Command.LoadImages(
      modelId,
      versionFilter.now(),
      sort.now(),
      includeNsfw.now()
    )
    if (!service.currentImageQuery.contains(query)) service.push(query)
  }

  lazy val element: HtmlElement = div(
    onMountCallback(_ => load()),
    includeNsfw.signal.changes --> (_ => load()),
    documentEvents(_.onKeyDown) --> { event =>
      if (event.key == "Escape") viewing.set(None)
    },
    toolbar,
    div(
      cls := "civitai-results-grid civitai-gallery-grid",
      children <-- items.split(_.key)((_, item, _) => tile(item))
    ),
    // Under the grid, what the search is doing: loading, failed — Retry asks
    // for exactly the page that failed — or able to bring ten more.
    child <-- Signal
      .combine(
        items.map(_.isEmpty),
        service.loadingImages,
        service.imageFailure,
        service.hasMoreImages
      )
      .map {
        case (_, true, _, _) =>
          p(
            cls := "text-secondary mt-3 is-flex is-align-items-center",
            span(cls := "media-loader is-inline"),
            "Loading posted images..."
          )
        case (_, _, Some(reason), _) =>
          RetryNotice(
            s"The posted images did not load. $reason",
            () => service.retryImages()
          ).element
        case (_, _, _, true) =>
          button(
            cls := "button is-outlined is-fullwidth mt-3",
            "Load More",
            onClick --> (_ =>
              service.push(CivitaiService.Command.LoadMoreImages)
            )
          )
        case (true, _, _, _) =>
          p(cls := "text-secondary mt-3", "No images for this version.")
        case _ => emptyNode
      },
    child.maybe <-- viewing.signal.map(_.map(viewer))
  )

  private def toolbar: HtmlElement = div(
    cls := "field is-grouped is-grouped-multiline is-align-items-center mb-3",
    div(
      cls := "control",
      div(
        cls := "select is-small",
        select(
          // The search payload's synthetic "default" version has id 0.
          children <-- versions.map(
            _.filter(_.id != 0).map(version =>
              option(
                value := version.id.toString,
                version.name +
                  version.baseModel.fold("")(base => s" · $base"),
                selected <-- versionFilter.signal.map(_.contains(version.id))
              )
            )
          ),
          onChange.mapToValue --> { chosen =>
            versionFilter.set(chosen.toIntOption)
            load()
          }
        )
      )
    ),
    div(
      cls := "control",
      div(
        cls := "select is-small",
        select(
          Sorts.map(order =>
            option(value := order, order, selected := order == sort.now())
          ),
          onChange.mapToValue --> { order =>
            sort.set(order)
            load()
          }
        )
      )
    ),
    label(
      cls := "checkbox is-size-7 control",
      input(
        typ := "checkbox",
        cls := "mr-1",
        controlled(
          checked <-- includeNsfw.signal,
          onClick.mapToChecked --> includeNsfw
        )
      ),
      "Include NSFW"
    )
    // Play videos is the browser's own switch, in the modal's head.
  )

  private def describe(item: GalleryItem): String =
    item.username.fold("An image of this version")(user => s"Posted by $user")

  private def tile(item: GalleryItem): HtmlElement = div(
    cls := "civitai-card cursor-pointer",
    onClick --> (_ => viewing.set(Some(item))),
    BrowserMedia(
      List(ModelExample(item.url, item.video)),
      describe(item)
    ).element
  )

  /** The image at full size over everything, with who posted it and the way to
    * its page; a click outside the media or Esc closes it. What the tile
    * already loaded shows at once — the image's small copy, the video's still,
    * both in the browser's cache — and the larger file takes over once it is
    * in.
    */
  private def viewer(item: GalleryItem): HtmlElement = {
    val loaded = Var(false)
    div(
      cls := "media-lightbox",
      // The browser modal's body is right behind: it must not scroll either.
      ScrollLock.whileMounted,
      onClick --> (_ => viewing.set(None)),
      div(
        cls := "media-lightbox-frame",
        if (item.video)
          videoTag(
            cls := "media-lightbox-media",
            src := item.url,
            VideoAttrs.poster(BrowserUtils.civitaiStill(item.url)),
            VideoAttrs.controls,
            VideoAttrs.autoplay,
            VideoAttrs.muted,
            VideoAttrs.loop,
            VideoAttrs.playsInline,
            VideoAttrs.onLoadedData --> (_ => loaded.set(true)),
            onClick.stopPropagation --> (_ => ())
          )
        else
          img(
            cls := "media-lightbox-media",
            src <-- loaded.signal.map(full =>
              if (full) item.url
              else BrowserUtils.civitaiThumbnail(item.url, video = false)
            ),
            alt := describe(item),
            onClick.stopPropagation --> (_ => ())
          ),
        // The original loads out of sight, then takes the small copy's place.
        Option.when(!item.video)(
          img(
            src := item.url,
            styleAttr := "display: none",
            onLoad --> (_ => loaded.set(true))
          )
        ),
        child.maybe <-- loaded.signal.map(done =>
          Option.when(!done)(div(cls := "media-loader"))
        )
      ),
      div(
        cls := "media-lightbox-caption",
        item.username.map(user => span(s"by $user")),
        a(
          href := item.page,
          target := "_blank",
          rel := "noopener noreferrer",
          onClick.stopPropagation --> (_ => ()),
          "Open on Civitai ↗"
        )
      )
    )
  }
}

object CivitaiImageGallery {

  /** Civitai's image orders, the default first. */
  val Sorts: List[String] = List("Most Reactions", "Newest", "Most Comments")

  private val MediaId =
    "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}".r

  /** One tile — an image the author published with the version, or a post from
    * the image search — and its page on civitai.com.
    */
  case class GalleryItem(
      url: String,
      video: Boolean,
      username: Option[String],
      page: String
  ) {

    /** The image itself, however it arrived: its id on Civitai's CDN when the
      * URL carries one, so an image the search returns again shows once.
      */
    val key: String = MediaId.findFirstIn(url).getOrElse(url)
  }
}
