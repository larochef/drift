package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** The chips a Civitai version and its files carry in the browser
  * (`specs/24-model-details-in-browsers.md`).
  */
object CivitaiTags {

  /** The version's base model, as a chip beside its name. A model's versions
    * can target different base models, and the search filter does not reach
    * into the detail view — so a version for another base model than the
    * architecture being browsed (`civitaiBaseModels`) gets a warning, but stays
    * installable: it is recorded for this architecture either way, and the
    * runtime will say if it truly cannot work.
    */
  def baseModel(
      version: CivitaiModelVersion,
      civitaiBaseModels: List[String]
  ): Option[HtmlElement] =
    version.baseModel.map { base =>
      if (civitaiBaseModels.isEmpty || civitaiBaseModels.contains(base))
        span(cls := "tag is-small ml-2", base)
      else
        span(
          cls := "tag is-warning is-small ml-2",
          title := s"This version was published for '$base', not one of " +
            "this architecture's base models " +
            s"(${civitaiBaseModels.mkString(", ")}). It still installs for " +
            "this architecture; if it misbehaves at runtime, this is why.",
          s"⚠ $base"
        )
    }

  /** What Civitai charges for this version, when it charges. Per **version**: a
    * model commonly publishes a paid one beside free ones, so the chip sits
    * with the version in the Files tab rather than on the tile, where it could
    * only say "something here is paid" (François, 2026-09-18).
    *
    * drift cannot buy it: a download of a paid version answers 401/403 unless
    * the account behind the Civitai token in Settings already owns it. Early
    * access says until when — after that date the same file downloads freely.
    */
  def paidAccess(version: CivitaiModelVersion): Option[HtmlElement] =
    version.paidAccess.map { access =>
      val until = version.paidUntil.map(CivitaiTags.dayOf)
      span(
        cls := "tag is-warning is-small ml-2",
        title := (if (access.permanent)
                    "Civitai charges for this version. It downloads only for " +
                      "an account that has bought it — the one whose API " +
                      "token is in Settings."
                  else
                    "Early access: Civitai charges for this version until " +
                      until.getOrElse("a date it does not state") +
                      ". Until then it downloads only for an account that " +
                      "has bought it; after, for anyone."),
        until match {
          case Some(date) => s"💲 early access until $date"
          case None       => "💲 paid"
        }
      )
    }

  /** The day of an ISO timestamp, as Civitai writes it — "2026-09-30" out of
    * "2026-09-30T18:10:41.893Z". Anything else is shown as it came.
    */
  def dayOf(timestamp: String): String = timestamp.takeWhile(_ != 'T')

  /** Precision/variant chips for one file — a version can publish the *same
    * filename* in several precisions (seen live: int8 and bf16), so without
    * these there is no way to tell which file is which. int8 additionally gets
    * a warning: sd-cpp has no INT8 matmul kernel on Vulkan or ROCm (upstream
    * issue #1929, `docs/int8_convrot.md`), so those checkpoints generate on the
    * CPU, 50-100x slower. fp8, scaled or not, stays on the GPU: without an fp8
    * kernel sd-cpp casts each layer's weights to bf16 as it computes.
    */
  def ofFile(file: CivitaiModelFile, downloaded: Boolean): Seq[HtmlElement] = {
    val quant = file.quantization
    val cpuBound = quant.contains("int8")
    Seq(
      quant.map(q =>
        span(
          cls := "tag is-info is-small ml-2",
          title := "Precision / quantization of this file",
          q
        )
      ),
      file.sizeVariant.map(variant =>
        span(cls := "tag is-small ml-1", variant)
      ),
      Option.when(cpuBound)(
        span(
          cls := "tag is-warning is-small ml-1",
          title := "sd-cpp has no INT8 matmul kernel on Vulkan or ROCm (issue " +
            "#1929): this file will generate on the CPU, 50-100x slower. " +
            "Prefer a bf16, fp8 or GGUF variant, or convert it.",
          "⚠ runs on CPU"
        )
      ),
      Option.when(quant.contains("nvfp4"))(
        span(
          cls := "tag is-warning is-small ml-1",
          title := "NVFP4 is NVIDIA's Blackwell FP4 format; it will not run " +
            "on AMD hardware.",
          "⚠ nvidia only"
        )
      ),
      Option.when(downloaded)(
        span(
          cls := "tag is-success is-small ml-1",
          title := "This file is already in the model cache",
          "✓ downloaded"
        )
      )
    ).flatten
  }
}
