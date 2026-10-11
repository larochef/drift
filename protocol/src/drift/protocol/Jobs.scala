package drift.protocol

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.*

/** Where a server takes its jobs, and which protocol this is. */
object JobRoutes {

  /** Raised when a request or an answer changes in a way the other side must
    * know. The drift runner says it in its capabilities; sd-server, whose
    * protocol this follows, says none.
    */
  val Version = 1

  val Root = "/sdcpp/v1"
  val Capabilities = s"$Root/capabilities"
  val ImageJobs = s"$Root/img_gen"
  val VideoJobs = s"$Root/vid_gen"
  val UpscaleJobs = s"$Root/upscale"
  val Jobs = s"$Root/jobs"

  def job(id: String): String = s"$Jobs/$id"
  def cancel(id: String): String = s"$Jobs/$id/cancel"
}

/** The native sdcpp job document, as `GET /sdcpp/v1/jobs/{id}` answers it.
  * Timestamps are unix seconds; drift converts to millis at the boundary.
  * `b64_json` is pinned by annotation: `enforce_snake_case` would split the
  * digit boundary into `b_64_json` and silently miss the field.
  */
case class NativeJobImage(
    index: Int = 0,
    @named("b64_json") b64Json: String = ""
)
case class NativeJobError(
    code: String = "",
    message: String = ""
)
case class NativeJobResult(
    outputFormat: String = "",
    mimeType: Option[String] = None,
    fps: Option[Int] = None,
    frameCount: Option[Int] = None,
    images: List[NativeJobImage] = List.empty,
    @named("b64_json") b64Json: Option[String] = None,
    /** A video's upscale on the runner: the file it wrote, and how many frames
      * its source has (`frameCount` of them are in the file).
      */
    file: Option[String] = None,
    sourceFrames: Option[Int] = None,
    /** The exact rate of such a file, as a fraction or a number (`fps` is it
      * rounded).
      */
    frameRate: Option[String] = None
)

/** The picture a job is sampling, in a job whose request asked for one
  * (`preview`, `specs/54-step-preview.md`), as sd-server and the drift runner
  * both give it: the sampling pass, from 1 (an image of the batch, a video's
  * expert; on sd-cpp a hires pass counts too), the step within it, and a PNG, a
  * JPEG or, for a video on the drift runner, a GIF that plays it.
  */
case class NativeJobPreview(
    pass: Int = 0,
    step: Int = 0,
    totalSteps: Int = 0,
    @named("b64_json") b64Json: String = ""
) {

  /** The picture's type, told from how its base64 starts: the drift runner's
    * JPEG, its GIF of a video, else sd-server's PNG.
    */
  def mimeType: String =
    if (b64Json.startsWith("/9j/")) "image/jpeg"
    else if (b64Json.startsWith("R0lGOD")) "image/gif"
    else "image/png"
}
case class NativeJob(
    id: String = "",
    kind: String = "",
    status: String = "",
    created: Long = 0,
    started: Option[Long] = None,
    completed: Option[Long] = None,
    queuePosition: Int = 0,
    result: Option[NativeJobResult] = None,
    error: Option[NativeJobError] = None,
    preview: Option[NativeJobPreview] = None,
    /** What the job went on without (the drift runner's: a LoRA left
      * unapplied).
      */
    warnings: List[String] = Nil,
    /** Where to ask for the job again, in the answer to its submission. */
    pollUrl: Option[String] = None
)

object NativeJob {

  /** Written with every field that has a value, one equal to its default
    * included — a queue position of 0 is still said, as sd-server says it — and
    * read with the defaults for what a server leaves out.
    */
  given JsonValueCodec[NativeJob] = JsonCodecMaker.make(
    CodecMakerConfig
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
      .withTransientDefault(false)
      .withTransientEmpty(false)
  )
}

/** What drift's runner says of itself in its capabilities document, under
  * `drift`; sd-server says none of it.
  */
case class ServerFacts(protocol: Int)

/** The part of a capabilities document that tells which server answers. */
case class CapabilitiesHead(drift: Option[ServerFacts] = None)
object CapabilitiesHead {
  given JsonValueCodec[CapabilitiesHead] = JsonCodecMaker.make
}
