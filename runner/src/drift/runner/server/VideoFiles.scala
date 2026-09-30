package drift.runner.server

import drift.runner.diffusion.Video

import java.io.{BufferedOutputStream, IOException}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.Files

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
