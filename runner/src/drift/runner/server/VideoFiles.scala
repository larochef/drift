package drift.runner.server

import drift.runner.diffusion.*

import java.awt.image.BufferedImage
import java.io.*
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO

/** A video as the file drift saves: a webm (VP8, and Vorbis for a soundtrack),
  * encoded by the `ffmpeg` on the `PATH`. sd-server links libvpx and libwebm
  * itself; the runner leaves both to ffmpeg, a dependency of the runner's video
  * models (`specs/42`, step 14).
  */
object VideoFiles {

  def webm(video: Video): Array[Byte] = {
    require(video.frames.nonEmpty, "a video without frames")
    val (width, height) =
      (video.frames.head.getWidth, video.frames.head.getHeight)
    val folder = Files.createTempDirectory("drift-video")
    try {
      val output = folder.resolve("video.webm")
      val soundtrack = video.soundtrack.map { track =>
        val wav = folder.resolve("soundtrack.wav")
        Files.write(wav, wave(track.samples, track.channels, track.rate))
        wav
      }
      val command =
        Seq("ffmpeg", "-hide_banner", "-loglevel", "error", "-y") ++
          Seq(
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            s"${width}x$height"
          ) ++
          Seq("-r", video.fps.toString, "-i", "-") ++
          soundtrack.toSeq.flatMap(wav => Seq("-i", wav.toString)) ++
          Seq(
            "-c:v",
            "libvpx",
            "-crf",
            "6",
            "-b:v",
            "40M",
            "-pix_fmt",
            "yuv420p"
          ) ++
          soundtrack.toSeq.flatMap(_ =>
            Seq("-c:a", "libvorbis", "-shortest")
          ) ++
          Seq(output.toString)
      val log = folder.resolve("ffmpeg.log")
      val process =
        try
          new ProcessBuilder(command*)
            .redirectErrorStream(true)
            .redirectOutput(log.toFile)
            .start()
        catch {
          case error: IOException =>
            throw new IllegalStateException(
              s"the drift runner encodes videos with ffmpeg, which is not installed: ${error.getMessage}"
            )
        }
      val in = new BufferedOutputStream(process.getOutputStream, 1 << 20)
      try {
        val row = new Array[Byte](width * 3)
        video.frames.foreach { frame =>
          (0 until height).foreach { y =>
            (0 until width).foreach { x =>
              val rgb = frame.getRGB(x, y)
              row(3 * x) = (rgb >> 16).toByte
              row(3 * x + 1) = (rgb >> 8).toByte
              row(3 * x + 2) = rgb.toByte
            }
            in.write(row)
          }
        }
      } finally in.close()
      val status = process.waitFor()
      if (status != 0)
        throw new IllegalStateException(
          s"ffmpeg failed ($status): ${Files.readString(log).trim}"
        )
      Files.readAllBytes(output)
    } finally
      Files
        .walk(folder)
        .sorted(java.util.Comparator.reverseOrder())
        .forEach(path => Files.deleteIfExists(path))
  }

  /** The largest short edge a decoded clip keeps: every video model scales its
    * inputs to its own canvas, and full-HD frames of a 15-second clip already
    * take gigabytes of heap.
    */
  private val ClipShortEdge = 1080

  /** The largest short edge a decoded still keeps (MiniMax H3 encodes a
    * reference image at 2048).
    */
  private val StillShortEdge = 2048

  /** An uploaded media file, decoded: an image the JDK reads directly is a
    * still; anything else goes through `ffprobe` and `ffmpeg`, a single frame
    * without sound being a still too, frames a clip (with its soundtrack, at
    * most stereo, at its own rate), and sound alone a sound.
    */
  def decode(bytes: Array[Byte]): Media =
    Option(ImageIO.read(new ByteArrayInputStream(bytes))) match {
      case Some(image) => Media.Still(image)
      case None        =>
        val folder = Files.createTempDirectory("drift-media")
        try {
          val file = folder.resolve("media")
          Files.write(file, bytes)
          decodeFile(file)
        } finally
          Files
            .walk(folder)
            .sorted(java.util.Comparator.reverseOrder())
            .forEach(path => Files.deleteIfExists(path))
    }

  private def decodeFile(file: Path): Media = {
    val probe = ujson.read(
      run(
        Seq(
          "ffprobe",
          "-v",
          "error",
          "-show_entries",
          "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,channels,sample_rate",
          "-of",
          "json",
          file.toString
        )
      )
    )
    val streams = probe("streams").arr.toSeq
    def stream(kind: String) =
      streams.find(_.obj.get("codec_type").exists(_.str == kind))
    val soundtrack = stream("audio").map { audio =>
      val channels = math.min(audio("channels").num.toInt, 2)
      val rate = audio("sample_rate").str.toInt
      val raw = run(
        Seq("ffmpeg", "-hide_banner", "-loglevel", "error", "-i") ++
          Seq(file.toString, "-vn", "-f", "f32le", "-ac", channels.toString) ++
          Seq("-ar", rate.toString, "-")
      )
      val buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
      val samples = new Array[Float](raw.length / 4)
      buffer.asFloatBuffer().get(samples)
      Soundtrack(samples, channels, rate)
    }
    stream("video") match {
      case None =>
        Media.Sound(
          soundtrack.getOrElse(
            throw new IllegalArgumentException(
              "a media file with neither frames nor sound"
            )
          )
        )
      case Some(video) =>
        val (width, height) =
          (video("width").num.toInt, video("height").num.toInt)
        val scale =
          math.min(1.0, ClipShortEdge.toDouble / math.min(width, height))
        def even(side: Double) = math.max(2, (math.round(side) / 2 * 2).toInt)
        val (w, h) =
          if (scale < 1) (even(width * scale), even(height * scale))
          else (width, height)
        val raw = run(
          Seq("ffmpeg", "-hide_banner", "-loglevel", "error", "-i") ++
            Seq(file.toString, "-an", "-vf", s"scale=$w:$h:flags=lanczos") ++
            Seq("-f", "rawvideo", "-pix_fmt", "rgb24", "-")
        )
        val frameBytes = w * h * 3
        val frames = (0 until raw.length / frameBytes).map { index =>
          val image = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
          val base = index * frameBytes
          (0 until h).foreach { y =>
            (0 until w).foreach { x =>
              val at = base + 3 * (y * w + x)
              image.setRGB(
                x,
                y,
                (raw(at) & 0xff) << 16 | (raw(at + 1) & 0xff) << 8 | raw(
                  at + 2
                ) & 0xff
              )
            }
          }
          image
        }
        if (frames.isEmpty)
          throw new IllegalArgumentException("a video without frames")
        if (frames.size == 1 && soundtrack.isEmpty) {
          // A still ffmpeg reads and the JDK doesn't (WebP): decoded again
          // at the stills' larger bound.
          val stillScale =
            math.min(1.0, StillShortEdge.toDouble / math.min(width, height))
          if (stillScale == scale) Media.Still(frames.head)
          else {
            val (sw, sh) = (even(width * stillScale), even(height * stillScale))
            val png = run(
              Seq("ffmpeg", "-hide_banner", "-loglevel", "error", "-i") ++
                Seq(file.toString, "-vf", s"scale=$sw:$sh:flags=lanczos") ++
                Seq("-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-")
            )
            Media.Still(ImageIO.read(new ByteArrayInputStream(png)))
          }
        } else {
          def rate(value: String) = value.split('/') match {
            case Array(n, d) if d.toDouble > 0 => n.toDouble / d.toDouble
            case _                             => 0.0
          }
          val fps = Seq("avg_frame_rate", "r_frame_rate")
            .flatMap(video.obj.get)
            .map(v => rate(v.str))
            .find(_ > 0)
            .getOrElse(24.0)
          Media.Clip(frames, fps, soundtrack)
        }
    }
  }

  /** `command`'s standard output; its errors when it fails. */
  private def run(command: Seq[String]): Array[Byte] = {
    val process =
      try new ProcessBuilder(command*).start()
      catch {
        case error: IOException =>
          throw new IllegalStateException(
            s"the drift runner decodes media with ${command.head}, which is not installed: ${error.getMessage}"
          )
      }
    val errorBytes = new java.io.ByteArrayOutputStream()
    val drain = new Thread(() => {
      process.getErrorStream.transferTo(errorBytes); ()
    })
    drain.start()
    val out = new BufferedInputStream(process.getInputStream).readAllBytes()
    drain.join()
    val status = process.waitFor()
    if (status != 0)
      throw new IllegalArgumentException(
        s"${command.head} could not read the media ($status): ${errorBytes.toString.trim}"
      )
    out
  }

  /** 16-bit PCM WAV of interleaved `samples` in [−1, 1]. */
  private def wave(
      samples: Array[Float],
      channels: Int,
      rate: Int
  ): Array[Byte] = {
    val data = samples.length * 2
    val buffer = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
    buffer.put("RIFF".getBytes).putInt(36 + data).put("WAVE".getBytes)
    buffer
      .put("fmt ".getBytes)
      .putInt(16)
      .putShort(1)
      .putShort(channels.toShort)
    buffer.putInt(rate).putInt(rate * channels * 2)
    buffer.putShort((channels * 2).toShort).putShort(16)
    buffer.put("data".getBytes).putInt(data)
    samples.foreach(sample =>
      buffer.putShort(
        math.round(math.max(-1f, math.min(1f, sample)) * 32767f).toShort
      )
    )
    buffer.array()
  }
}
