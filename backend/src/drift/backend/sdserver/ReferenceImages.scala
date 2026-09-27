package drift.backend.sdserver

import drift.backend.postprocess.PostProcessImages
import drift.backend.runtime.SdCppBuilds
import drift.shared.*

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.util.Base64
import javax.imageio.ImageIO
import scala.util.control.NonFatal

/** A generation's reference images as sd-server should get them, so they reach
  * the model the way sd-cli would hand them over — their own shape, about a
  * megapixel.
  *
  * sd-server's `auto_resize_ref_image` decides two resizes at once. With the
  * default, it stretches every reference to the request's width × height while
  * decoding the request — a portrait reference beside a square output came out
  * half again as wide — and scales it again before the VAE, to about a
  * megapixel with its (by then stretched) shape. With `false` it does neither,
  * and a large reference would go to the VAE at full size.
  *
  * So on a build that honours `false` (`SdCppBuilds.keepsReferenceSize`) drift
  * does the second resize itself, as sd-cpp's `resize_before_vae` would, and
  * asks for neither; on an older one, whose sd-server stretches whatever it is
  * told, each reference is letterboxed to the output's shape with neutral grey
  * so the stretch is an even scale.
  */
private[sdserver] object ReferenceImages {

  def prepared(
      parameters: ImageGenerationParameters,
      architecture: Architecture,
      runtime: Runtime
  ): ImageGenerationParameters =
    if (parameters.refImages.isEmpty) parameters
    else if (SdCppBuilds.keepsReferenceSize(runtime))
      parameters.copy(
        refImages = parameters.refImages.map(
          reshaped(_)(image =>
            scaledForVae(
              image,
              parameters.width,
              parameters.height,
              architecture
            )
          )
        ),
        autoResizeRefImage = Some(false)
      )
    else
      parameters.copy(refImages =
        parameters.refImages.map(
          reshaped(_)(
            PostProcessImages
              .letterboxed(_, parameters.width, parameters.height)
          )
        )
      )

  /** The size sd-cpp's `resize_before_vae` gives a reference
    * (`src/pipeline/image.cpp`): an area of a megapixel — the output's own for
    * Qwen-Image 2.1 — never more than the output's, the reference's shape kept,
    * each side rounded to 32 for the Qwen-Image family and 16 for the rest.
    * sd-cpp tells the families apart by its own version enum; drift by the
    * built-in architecture ids.
    */
  def vaeSizeOf(
      referenceWidth: Int,
      referenceHeight: Int,
      outputWidth: Int,
      outputHeight: Int,
      architecture: Architecture
  ): (Int, Int) = {
    val outputArea = outputWidth.toLong * outputHeight
    val wanted =
      if (architecture.id == "qwen-image-2.1") outputArea else 1024L * 1024
    val area = math.min(wanted, outputArea).toDouble
    val width = math.sqrt(area * referenceWidth / referenceHeight)
    val height = width * referenceHeight / referenceWidth
    val factor = if (architecture.id.startsWith("qwen-image")) 32 else 16
    def rounded(side: Double) =
      (math.round(side / factor) * factor).toInt.max(factor)
    (rounded(width), rounded(height))
  }

  private def scaledForVae(
      image: BufferedImage,
      outputWidth: Int,
      outputHeight: Int,
      architecture: Architecture
  ): BufferedImage = {
    val (width, height) = vaeSizeOf(
      image.getWidth,
      image.getHeight,
      outputWidth,
      outputHeight,
      architecture
    )
    PostProcessImages.scaledCopy(image, width, height)
  }

  /** `reference` — a data URL or bare base64 — put through `change` and sent
    * back as a PNG data URL; left as it was when it does not decode, since
    * sd-server gives the clearer refusal.
    */
  private def reshaped(
      reference: String
  )(change: BufferedImage => BufferedImage): String =
    try {
      val encoded =
        if (reference.startsWith("data:"))
          reference.drop(reference.indexOf(',') + 1)
        else reference
      Option(
        ImageIO.read(
          ByteArrayInputStream(Base64.getMimeDecoder.decode(encoded))
        )
      ).fold(reference)(image => PostProcessImages.dataUrl(change(image)))
    } catch { case NonFatal(_) => reference }
}
