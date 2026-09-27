package drift.frontend.pages.cache

import drift.frontend.components.*
import drift.frontend.services.{BrowserServices, UpscalerService}
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** The upscaler store (`specs/10-generation-time-upscaling.md`): the classic
  * RealESRGAN weights one click away, a free-form URL for anything else, and
  * Civitai's `Upscaler` category through the existing browser. Global, not
  * per-architecture — an ESRGAN model upscales pixels, whatever produced them.
  */
class UpscalerSection(
    browsers: BrowserServices,
    service: UpscalerService
) extends Component {
  private val browserOpen = Var(false)
  private val urlVar = Var("")

  private def installedIds(upscalers: List[Upscaler]): Set[String] =
    upscalers.map(_.id).toSet

  private def jobTag(job: UpscalerDownloadJob): HtmlElement = job.state match {
    case DownloadState.Downloading =>
      val percent = job.totalBytes
        .filter(_ > 0)
        .map(total => s" ${(job.downloadedBytes * 100 / total).min(100)}%")
        .getOrElse("…")
      span(cls := "tag is-info is-small", s"⬇$percent")
    case DownloadState.Queued =>
      span(cls := "tag is-info is-small", "⬇ queued")
    case DownloadState.Failed =>
      span(cls := "tag is-danger is-small", "✗ failed")
    case DownloadState.Cancelled =>
      span(cls := "tag is-small", "cancelled")
    case DownloadState.Completed =>
      span(
        cls := "tag is-success is-small",
        s"✓ ${LaunchBlocker.humanBytes(job.downloadedBytes)}"
      )
  }

  private def curatedRow(
      entry: UpscalerSection.CuratedUpscaler,
      installed: Set[String]
  ): HtmlElement = {
    val alreadyInstalled = installed.contains(entry.stem)
    div(
      cls := "level is-mobile mb-1 is-marginless",
      div(
        cls := "level-left",
        div(
          p(cls := "text-primary is-size-7", entry.label),
          p(cls := "text-secondary is-size-7", entry.note)
        )
      ),
      div(
        cls := "level-right",
        if (alreadyInstalled)
          span(cls := "tag is-success is-small", "installed")
        else
          button(
            cls := "button is-info is-small",
            "⬇ Install",
            onClick --> (_ =>
              service.push(
                UpscalerService.Command.InstallUrl(
                  url = entry.url,
                  label = Some(entry.label),
                  fileName = Some(entry.fileName),
                  sha256 = None
                )
              )
            )
          )
      )
    )
  }

  private def installedRow(
      upscaler: Upscaler,
      jobs: List[UpscalerDownloadJob]
  ): HtmlElement = {
    val job = jobs.find(_.upscalerId == upscaler.id)
    div(
      cls := "level is-mobile mb-1 is-marginless",
      div(
        cls := "level-left",
        div(
          p(cls := "text-primary is-size-7", upscaler.label),
          p(
            cls := "text-secondary is-size-7",
            // The stem is what the generation form's dropdown will show.
            s"name: ${upscaler.id}"
          )
        )
      ),
      div(
        cls := "level-right",
        job.map(j => span(cls := "mr-2", jobTag(j))),
        button(
          cls := "button is-danger is-small",
          "🗑️",
          title := "Delete the upscaler and its weight file",
          onClick --> (_ =>
            if (
              window.confirm(
                s"Delete upscaler '${upscaler.label}' and its file?"
              )
            ) service.push(UpscalerService.Command.Delete(upscaler.id))
          )
        )
      )
    )
  }

  private def handleUrlInstall(): Unit = {
    val url = urlVar.now().trim
    if (url.nonEmpty) {
      service.push(
        UpscalerService.Command.InstallUrl(url, None, None, None)
      )
      urlVar.set("")
    }
  }

  lazy val element: HtmlElement = div(
    p(
      cls := "text-secondary is-size-7",
      "Upscalers live in ~/.cache/drift/upscale and are passed to every " +
        "session as --hires-upscalers-dir. sd-server scans the directory at " +
        "launch, so a session already running needs a restart to see a new " +
        "one."
    ),
    div(
      cls := "box bg-table-header py-2 px-3 mb-3",
      label(cls := "label text-primary is-small mb-1", "RealESRGAN classics"),
      children <-- service.upscalers.map { upscalers =>
        val installed = installedIds(upscalers)
        UpscalerSection.Curated.map(curatedRow(_, installed))
      }
    ),
    div(
      cls := "box bg-table-header py-2 px-3 mb-3",
      label(
        cls := "label text-primary is-small mb-1",
        "From a URL or Civitai"
      ),
      div(
        cls := "field has-addons mb-1",
        div(
          cls := "control is-expanded",
          input(
            cls := "input is-small",
            placeholder := "https://… direct link to a .pth / .safetensors upscaler",
            value <-- urlVar.signal,
            onInput.mapToValue --> urlVar
          )
        ),
        div(
          cls := "control",
          button(
            cls := "button is-info is-small",
            "⬇ Install",
            disabled <-- urlVar.signal.map(_.trim.isEmpty),
            onClick --> (_ => handleUrlInstall())
          )
        )
      ),
      button(
        cls := "button is-small is-info",
        span(cls := "plus-icon", "+"),
        " Browse Civitai upscalers",
        onClick --> (_ => browserOpen.set(true))
      )
    ),
    label(cls := "label text-primary is-small mb-1", "Installed"),
    children <-- service.upscalers.combineWith(service.jobs).map {
      (upscalers, jobs) =>
        if (upscalers.isEmpty)
          List(
            p(
              cls := "text-secondary is-size-7",
              "No upscalers installed yet."
            )
          )
        else upscalers.map(installedRow(_, jobs))
    },
    child <-- browserOpen.signal.map {
      case false => emptyNode
      case true  =>
        CivitaiBrowser(
          service = browsers.civitai,
          initialQuery = "",
          initialModelType = Some("UPSCALER"),
          civitaiBaseModels = List.empty,
          // Never called in version-select mode; the version button installs.
          onSelect = (_, _, _, _, _) => (),
          onCancel = () => browserOpen.set(false),
          // The browser stays open like the LoRA one: progress shows on this
          // page, and several upscalers often come from one search. Ticks may
          // cross versions; an upscaler is installed by version, so they go
          // out one version at a time.
          onInstall = Some { (civitaiModelId, files, _) =>
            files
              .groupBy(_.versionId)
              .foreach((versionId, chosen) =>
                service.push(
                  UpscalerService.Command.InstallCivitai(
                    civitaiModelId,
                    versionId,
                    chosen.map(_.fileId)
                  )
                )
              )
          },
          installed = service.upscalers.map(Installed.upscalers)
        ).element
    }
  )
}

object UpscalerSection {

  /** One entry of the curated list — the RRDBNet-style RealESRGAN releases this
    * sd-cpp build actually loads (no SRVGG "compact" variants), fetched from
    * their canonical GitHub release URLs.
    */
  case class CuratedUpscaler(
      label: String,
      fileName: String,
      url: String,
      note: String
  ) {
    def stem: String = fileName.stripSuffix(".pth")
  }

  val Curated: List[CuratedUpscaler] = List(
    CuratedUpscaler(
      label = "RealESRGAN x4plus",
      fileName = "RealESRGAN_x4plus.pth",
      url =
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
      note = "4× — the general-purpose default (64 MB)"
    ),
    CuratedUpscaler(
      label = "RealESRGAN x4plus anime 6B",
      fileName = "RealESRGAN_x4plus_anime_6B.pth",
      url =
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
      note = "4× — tuned for anime and illustration (18 MB)"
    ),
    CuratedUpscaler(
      label = "RealESRGAN x2plus",
      fileName = "RealESRGAN_x2plus.pth",
      url =
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
      note = "2× — when 4× is more than needed (64 MB)"
    ),
    CuratedUpscaler(
      label = "RealESRNet x4plus",
      fileName = "RealESRNet_x4plus.pth",
      url =
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
      note = "4× — the GAN-free variant, smoother but softer (64 MB)"
    )
  )
}
