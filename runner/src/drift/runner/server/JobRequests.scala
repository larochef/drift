package drift.runner.server

import drift.protocol.SampleRequest
import drift.runner.diffusion.*

import java.io.ByteArrayInputStream
import java.util.Base64
import javax.imageio.ImageIO

/** What reading a job's request shares between the kinds of job: its media
  * decoded, whether it asks for previews, and what it sets that goes unread.
  */
private[server] object JobRequests {

  /** What a request sets that the runner does not read: said in the job's
    * warnings rather than dropped. A field at the value that means "nothing
    * asked" is not one.
    */
  def unread(
      sample: SampleRequest,
      highNoise: Option[SampleRequest],
      clipSkip: Option[Int],
      outputFormat: Option[String],
      written: String,
      others: Seq[(String, Boolean)]
  ): Seq[String] = {
    val samples = sample +: highNoise.toSeq
    val set = Seq(
      "scheduler" -> samples.exists(
        _.scheduler.exists(name => name.nonEmpty && name != "default")
      ),
      "eta" -> samples.exists(_.eta.exists(_ != 0)),
      "shifted_timestep" -> samples.exists(_.shiftedTimestep.exists(_ != 0)),
      "img_cfg" -> samples.exists(_.guidance.exists(_.imgCfg.nonEmpty)),
      "slg" -> samples.exists(
        _.guidance.exists(_.slg.exists(_.scale.exists(_ != 0)))
      ),
      "clip_skip" -> clipSkip.exists(_ > 0),
      "output_format" -> outputFormat.exists(format =>
        format.nonEmpty && format != written
      )
    ) ++ others
    val names = set.collect { case (name, true) => s"`$name`" }
    Option
      .when(names.nonEmpty)(
        s"the drift runner does not read ${names.mkString(", ")}: " +
          (if (names.size == 1) "it was" else "they were") + " ignored"
      )
      .toSeq
  }

  /** Whether a request asks for step previews: sd-server's `preview`, a mode
    * (`proj`, `tae`, `vae`) or `none`. The runner has one way to draw one,
    * whatever the mode asked.
    */
  def previewAsked(preview: Option[String]): Boolean =
    preview.exists(_ != "none")

  /** An image sent as base64 or a data URL. */
  def decode(encoded: String): java.awt.image.BufferedImage = {
    val data =
      if (encoded.startsWith("data:")) encoded.drop(encoded.indexOf(',') + 1)
      else encoded
    Option(
      ImageIO.read(new ByteArrayInputStream(Base64.getMimeDecoder.decode(data)))
    ).getOrElse(
      throw new IllegalArgumentException("an image that does not decode")
    )
  }

  /** A media input (base64 or a data URL), decoded. */
  def media(encoded: String): Media =
    VideoFiles.decode(
      Base64.getMimeDecoder.decode(
        if (encoded.startsWith("data:")) encoded.drop(encoded.indexOf(',') + 1)
        else encoded
      )
    )
}
