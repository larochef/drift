package drift.frontend.pages.cache

import drift.frontend.Page
import drift.frontend.components.*
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Two views, because the two sets genuinely differ
  * (`specs/05-model-cache-and-downloads.md`):
  *
  *   - **On disk** — everything the cache holds, orphans included. Answers
  *     "what is using my disk?", and is where space is reclaimed.
  *   - **Configured** — every registered model grouped by architecture, dimmed
  *     with a download control when absent. Answers "what do I still need
  *     before I can run this?".
  *
  * They overlap only in "registered and downloaded". Beside them, what was
  * declared and never fetched, and the upscalers. Each view lives in its own
  * file (`OnDiskCacheView`, `ConfiguredModelsView`, `MissingModelsView`,
  * `UpscalerSection`).
  *
  * Every tab is a URL — `/model-cache/<view>`, and for the on-disk view
  * `/model-cache/on-disk/<cache>` — so a refresh or a link lands on it
  * (François, 2026-09-11).
  */
class ModelCachePage(
    cacheService: CacheService,
    modelService: ModelService,
    architectureService: ArchitectureService,
    runConfigurationService: RunConfigurationService,
    downloadService: DownloadService,
    loraService: LoraService,
    upscalerService: UpscalerService,
    conversionService: ConversionService,
    browsers: BrowserServices,
    /** The path after `/model-cache/`: the view, then the on-disk cache. */
    section: Signal[List[String]]
) extends Component {
  private enum View(val slug: String) derives CanEqual {
    case OnDisk extends View("on-disk")
    case Configured extends View("configured")
    case Missing extends View("not-downloaded")
    case Upscalers extends View("upscalers")
  }

  /** Set from the URL only: the tabs are links. */
  private val view = Var[View](View.OnDisk)

  /** Which cache the on-disk view shows; ordered as the tabs render, each named
    * in the URL by its lowercased label.
    */
  private val kindTab = Var[CachedFileKind](CachedFileKind.HuggingFace)
  private val kindTabs: List[(CachedFileKind, String)] = List(
    CachedFileKind.HuggingFace -> "HuggingFace",
    CachedFileKind.Civitai -> "Civitai",
    CachedFileKind.Local -> "Local",
    CachedFileKind.Lora -> "LoRAs"
  )

  private def viewUrl(target: View): String =
    s"${Page.ModelCache.path}/${target.slug}"

  private def kindUrl(name: String): String =
    s"${viewUrl(View.OnDisk)}/${name.toLowerCase}"

  /** The view and cache the URL names; anything unknown is the default. */
  private val followUrl: Seq[Modifier[HtmlElement]] = Seq(
    section.map(
      _.headOption
        .flatMap(slug => View.values.find(_.slug == slug))
        .getOrElse(View.OnDisk)
    ) --> view,
    section.map(
      _.lift(1)
        .flatMap(slug => kindTabs.find(_._2.toLowerCase == slug).map(_._1))
        .getOrElse(CachedFileKind.HuggingFace)
    ) --> kindTab
  )

  /** Registered models whose weights are not on disk — `Missing` only: `Broken`
    * is a local path that download cannot fix.
    */
  private val missingModels: Signal[List[Model]] =
    modelService.allModels
      .combineWith(cacheService.statuses)
      .map { (models, statuses) =>
        models
          .filter(model =>
            statuses.get(model.id).exists(_.state == CacheState.Missing)
          )
          .sortBy(_.label)
      }

  lazy val element: HtmlElement = div(
    cls := "content",
    cacheService.effects,
    // Composed effects: this brings modelService's and architectureService's
    // pipelines along. Mounting those two directly as well would run every
    // command twice; mounting neither drops run-configuration loads entirely.
    runConfigurationService.effects,
    downloadService.effects,
    loraService.effects,
    upscalerService.effects,
    conversionService.effects,
    // A finished conversion is a new file on disk and a new registered
    // model: both listings change.
    conversionService.events --> Observer {
      case ConversionService.Event.Finished(_) =>
        cacheService.push(CacheService.Command.Load)
        cacheService.push(CacheService.Command.LoadFiles)
        modelService.push(ModelService.Command.Load)
      case _ => ()
    },
    downloadService.events --> Observer {
      case DownloadService.Event.Finished(_) =>
        cacheService.push(CacheService.Command.Load)
        cacheService.push(CacheService.Command.LoadFiles)
    },
    // An adopted orphan is referenced now; without the reload its row would
    // keep offering adoption.
    loraService.events --> Observer { case LoraService.Event.Adopted(_) =>
      cacheService.push(CacheService.Command.LoadFiles)
    },
    // A model created from the on-disk view (an orphan assigned to a family)
    // or deleted from it (an assignment undone) changes what the listing
    // should say about that file.
    modelService.events --> Observer {
      case ModelService.Event.Created(_) | ModelService.Event.Deleted(_) =>
        cacheService.push(CacheService.Command.Load)
        cacheService.push(CacheService.Command.LoadFiles)
    },
    onMountCallback { _ =>
      cacheService.push(CacheService.Command.Load)
      cacheService.push(CacheService.Command.LoadFiles)
      modelService.push(ModelService.Command.Load)
      architectureService.push(ArchitectureService.Command.Load)
      runConfigurationService.push(RunConfigurationService.Command.Load)
      downloadService.push(DownloadService.Command.Load)
      upscalerService.push(UpscalerService.Command.Load)
      upscalerService.push(UpscalerService.Command.LoadJobs)
      conversionService.push(ConversionService.Command.Load)
    },
    h1(cls := "title text-primary", "Model Cache"),
    ErrorBanner(cacheService),
    ErrorBanner(downloadService),
    ErrorBanner(loraService),
    ErrorBanner(upscalerService),
    ErrorBanner(conversionService),
    // Conversions above the tabs: they are started from the on-disk view
    // but their result matters to every view.
    ConversionJobList(conversionService).element,
    followUrl,
    div(
      cls := "tabs",
      ul(
        li(
          cls("is-active") <-- view.signal.map(_ == View.OnDisk),
          a("On disk", href := viewUrl(View.OnDisk))
        ),
        li(
          cls("is-active") <-- view.signal.map(_ == View.Configured),
          a("Configured", href := viewUrl(View.Configured))
        ),
        li(
          cls("is-active") <-- view.signal.map(_ == View.Missing),
          a(
            child.text <-- missingModels.map(missing =>
              s"Not downloaded (${missing.size})"
            ),
            href := viewUrl(View.Missing)
          )
        ),
        li(
          cls("is-active") <-- view.signal.map(_ == View.Upscalers),
          a("Upscalers", href := viewUrl(View.Upscalers))
        )
      )
    ),
    // The URL sets the view again when only the cache changes: rebuild on a
    // new view only.
    child <-- view.signal.distinct.map {
      case View.OnDisk =>
        OnDiskCacheView(
          kindTab.signal,
          kindTabs,
          kindUrl,
          cacheService,
          modelService,
          architectureService,
          runConfigurationService,
          loraService,
          conversionService
        ).element
      case View.Configured =>
        ConfiguredModelsView(
          architectureService,
          modelService,
          cacheService,
          downloadService
        ).element
      case View.Missing =>
        MissingModelsView(missingModels, cacheService, downloadService).element
      case View.Upscalers => UpscalerSection(browsers, upscalerService).element
    }
  )
}
