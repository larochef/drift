package drift.protocol

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}

/** The jobs a server is asked for, as they travel: sd-server's native protocol
  * (`/sdcpp/v1`), which drift's own runner speaks too.
  *
  * Every field is optional, because a server applies a request on top of what
  * it was launched with: a field left out means "the launch's". A server that
  * reads a request through these types refuses a field they do not name
  * (`Requests.read`) — nothing a client sends is dropped in silence — and the
  * backend's own requests are checked against them.
  */
case class SlgRequest(
    layers: Option[List[Int]],
    layerStart: Option[Double],
    layerEnd: Option[Double],
    scale: Option[Double]
)

case class GuidanceRequest(
    txtCfg: Option[Double],
    imgCfg: Option[Double],
    distilledGuidance: Option[Double],
    slg: Option[SlgRequest]
)

case class SampleRequest(
    scheduler: Option[String],
    sampleMethod: Option[String],
    sampleSteps: Option[Int],
    eta: Option[Double],
    shiftedTimestep: Option[Int],
    customSigmas: Option[List[Double]],
    flowShift: Option[Double],
    guidance: Option[GuidanceRequest]
) {

  /** The noise levels asked for; none when the list is absent or empty. */
  def sigmas: List[Double] = customSigmas.getOrElse(Nil)
}
object SampleRequest {

  /** A request that says nothing about sampling. */
  val none: SampleRequest =
    SampleRequest(None, None, None, None, None, None, None, None)
}

case class HiresRequest(
    enabled: Option[Boolean],
    upscaler: Option[String],
    scale: Option[Double],
    targetWidth: Option[Int],
    targetHeight: Option[Int],
    steps: Option[Int],
    denoisingStrength: Option[Double],
    customSigmas: Option[List[Double]],
    upscaleTileSize: Option[Int]
)

case class VaeTilingRequest(
    enabled: Option[Boolean],
    temporalTiling: Option[Boolean],
    tileSizeX: Option[Int],
    tileSizeY: Option[Int],
    targetOverlap: Option[Double],
    relSizeX: Option[Double],
    relSizeY: Option[Double],
    extraTilingArgs: Option[String]
)

case class LoraRequest(
    path: String,
    multiplier: Option[Double],
    isHighNoise: Option[Boolean]
)

/** `POST /sdcpp/v1/img_gen`. Pictures are base64, or data URLs. */
case class ImageJobRequest(
    prompt: Option[String],
    negativePrompt: Option[String],
    clipSkip: Option[Int],
    width: Option[Int],
    height: Option[Int],
    strength: Option[Double],
    seed: Option[Long],
    batchCount: Option[Int],
    initImage: Option[String],
    refImages: Option[List[String]],
    autoResizeRefImage: Option[Boolean],
    maskImage: Option[String],
    sampleParams: Option[SampleRequest],
    lora: Option[List[LoraRequest]],
    hires: Option[HiresRequest],
    vaeTilingParams: Option[VaeTilingRequest],
    outputFormat: Option[String],
    outputCompression: Option[Int],
    /** sd-server's step preview: a mode (`proj`, `tae`, `vae`) or `none`. */
    preview: Option[String]
)

case class VideoGuideRequest(media: String, frameIndex: Option[Int])

/** `POST /sdcpp/v1/vid_gen`. Media are base64, or data URLs. */
case class VideoJobRequest(
    prompt: Option[String],
    negativePrompt: Option[String],
    clipSkip: Option[Int],
    width: Option[Int],
    height: Option[Int],
    strength: Option[Double],
    seed: Option[Long],
    videoFrames: Option[Int],
    fps: Option[Int],
    moeBoundary: Option[Double],
    vaceStrength: Option[Double],
    initImage: Option[String],
    endImage: Option[String],
    controlFrames: Option[List[String]],
    references: Option[List[String]],
    guides: Option[List[VideoGuideRequest]],
    controlVideo: Option[String],
    controlStrength: Option[Double],
    controlStart: Option[Double],
    controlEnd: Option[Double],
    controlMask: Option[String],
    sourceVideo: Option[String],
    sampleParams: Option[SampleRequest],
    highNoiseSampleParams: Option[SampleRequest],
    lora: Option[List[LoraRequest]],
    vaeTilingParams: Option[VaeTilingRequest],
    outputFormat: Option[String],
    outputCompression: Option[Int],
    preview: Option[String],
    /** The drift runner's own, for a model with a soundtrack: the guidance on
      * the sound, and how far each of the picture and the sound follows the
      * other.
      */
    audioCfgScale: Option[Double],
    modalityScale: Option[Double],
    audioModalityScale: Option[Double]
)

/** `POST /sdcpp/v1/upscale`, the drift runner's own job: restored at
  * `width × height`, or at `scale` times the source's size.
  *
  * A picture is `source`, base64 or a data URL, and comes back as `img_gen`'s
  * result does. A video is a file, `sourceFile`, written to another,
  * `outputFile`: the server and its client share the machine, so a video of any
  * length goes through without being held.
  */
case class UpscaleJobRequest(
    source: Option[String],
    sourceFile: Option[String],
    outputFile: Option[String],
    scale: Option[Int],
    width: Option[Int],
    height: Option[Int],
    /** Negative, or absent: the server draws one. */
    seed: Option[Long],
    /** Frames restored at once, and frames two batches share. */
    batch: Option[Int],
    overlap: Option[Int]
)
object UpscaleJobRequest {

  def picture(source: String, scale: Int, seed: Long): UpscaleJobRequest =
    UpscaleJobRequest(
      source = Some(source),
      sourceFile = None,
      outputFile = None,
      scale = Some(scale),
      width = None,
      height = None,
      seed = Some(seed),
      batch = None,
      overlap = None
    )

  def video(
      sourceFile: String,
      outputFile: String,
      scale: Int,
      seed: Long
  ): UpscaleJobRequest =
    UpscaleJobRequest(
      source = None,
      sourceFile = Some(sourceFile),
      outputFile = Some(outputFile),
      scale = Some(scale),
      width = None,
      height = None,
      seed = Some(seed),
      batch = None,
      overlap = None
    )
}

object Requests {

  /** Snake case, as sd-server names its fields, and no field skipped: a request
    * naming one these types lack does not decode, and says which.
    */
  private inline def wire = CodecMakerConfig
    .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
    .withSkipUnexpectedFields(false)

  given JsonValueCodec[ImageJobRequest] = JsonCodecMaker.make(wire)
  given JsonValueCodec[VideoJobRequest] = JsonCodecMaker.make(wire)
  given JsonValueCodec[UpscaleJobRequest] = JsonCodecMaker.make(wire)
}
