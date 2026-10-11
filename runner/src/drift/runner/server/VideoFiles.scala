package drift.runner.server

import drift.runner.diffusion.*

import java.awt.image.{BufferedImage, DataBufferInt}
import java.io.*
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.*
import javax.imageio.ImageIO

/** A frame rate as a file states it, a fraction: 30000/1001 stays what it is
  * from the source to the video made of it, where a rounded 30 runs the frames
  * 0.1 % fast against their soundtrack.
  */
final case class FrameRate(numerator: Long, denominator: Long) {
  require(
    numerator > 0 && denominator > 0,
    s"a frame rate of $numerator/$denominator"
  )

  def value: Double = numerator.toDouble / denominator

  /** The nearest whole rate, for what only shows it. */
  def rounded: Int = math.max(1, math.round(value).toInt)

  /** As ffmpeg and ffprobe write it. */
  override def toString: String = s"$numerator/$denominator"
}

object FrameRate {

  def whole(fps: Int): FrameRate = FrameRate(fps, 1)

  /** ffprobe's `n/d`; none for its `0/0` of a stream without a rate. */
  def parse(text: String): Option[FrameRate] = text.split('/') match {
    case Array(n, d) =>
      for {
        numerator <- n.toLongOption
        denominator <- d.toLongOption
        if numerator > 0 && denominator > 0
      } yield FrameRate(numerator, denominator)
    case _ => None
  }
}

/** A file's frames as `ffprobe` describes them: their size, their rate and how
  * many there are — the stream's packets counted, which costs no decoding. A
  * few codecs carry packets that show nothing, so the count is what a progress
  * bar is drawn on, never what a loop ends on.
  */
final case class FramesProbe(
    width: Int,
    height: Int,
    rate: Option[FrameRate],
    count: Int
)

/** A file's sound as `ffprobe` describes it. */
final case class SoundProbe(codec: String, channels: Int, rate: Int)

/** What a media file holds. */
final case class MediaProbe(
    frames: Option[FramesProbe],
    sound: Option[SoundProbe]
)

/** A soundtrack the encoder takes from a file beside the frames it is sent:
  * `file`'s first sound stream, written as `codec` says (ffmpeg's output
  * options for it).
  */
final case class SoundFile(file: Path, codec: Seq[String])

/** A clip's frames, read from `ffmpeg` one at a time: a frame is on the heap
  * while its reader holds it and no longer, so a clip of any length is read in
  * the room of one frame. The frames are `width × height` at `rate`; `count` is
  * the probe's (`FramesProbe`). Closing it stops the decoder.
  */
final class ClipFrames private[server] (
    val width: Int,
    val height: Int,
    val rate: FrameRate,
    val count: Int,
    process: Process,
    log: Path
) extends Iterator[BufferedImage]
    with AutoCloseable {

  private val in = new BufferedInputStream(process.getInputStream, 1 << 20)
  private val bytes = new Array[Byte](width * height * 3)
  private var ahead: Option[BufferedImage] = None
  private var ended = false

  private def read(): Option[BufferedImage] =
    if (in.readNBytes(bytes, 0, bytes.length) < bytes.length) {
      val status = process.waitFor()
      val said = Files.readString(log).trim
      close()
      if (status != 0)
        throw new IllegalArgumentException(
          s"ffmpeg could not read the media ($status): $said"
        )
      None
    } else {
      val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
      val pixels =
        image.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
      var at = 0
      while (at < pixels.length) {
        pixels(at) = (bytes(3 * at) & 0xff) << 16 |
          (bytes(3 * at + 1) & 0xff) << 8 | bytes(3 * at + 2) & 0xff
        at += 1
      }
      Some(image)
    }

  def hasNext: Boolean = {
    if (ahead.isEmpty && !ended) ahead = read()
    ahead.isDefined
  }

  def next(): BufferedImage = {
    if (!hasNext) throw new NoSuchElementException("no frame left")
    val frame = ahead.get
    ahead = None
    frame
  }

  def close(): Unit = if (!ended) {
    ended = true
    process.destroyForcibly()
    try in.close()
    catch { case _: IOException => () }
    Files.deleteIfExists(log)
    ()
  }
}

/** A webm being written: `ffmpeg` runs from the first line, takes each frame as
  * it is made and writes `output` as it goes, so a video of any length is
  * encoded in the room of one frame, and what is written when the frames stop
  * coming is a video that plays. Frames are `width × height` at `rate`; `sound`
  * is muxed beside them, whatever it lasts: ffmpeg's `-shortest` would cut the
  * frames at a shorter soundtrack's end.
  */
final class WebmWriter private[server] (
    output: Path,
    width: Int,
    height: Int,
    rate: FrameRate,
    sound: Option[SoundFile]
) {
  private val log = Files.createTempFile("drift-video", ".log")
  private val command =
    Seq("ffmpeg", "-hide_banner", "-loglevel", "error", "-y") ++
      Seq("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", s"${width}x$height") ++
      Seq("-r", rate.toString, "-i", "-") ++
      sound.toSeq.flatMap(track => Seq("-i", track.file.toString)) ++
      // the sound's file may hold frames of its own: only ours are written
      Seq("-map", "0:v:0") ++ sound.toSeq.flatMap(_ => Seq("-map", "1:a:0")) ++
      Seq("-c:v", "libvpx", "-crf", "6", "-b:v", "40M") ++
      Seq("-pix_fmt", "yuv420p") ++
      sound.toSeq.flatMap(_.codec) ++
      Seq("-f", "webm", output.toString)
  private val process =
    try
      new ProcessBuilder(command*)
        .redirectErrorStream(true)
        .redirectOutput(log.toFile)
        .start()
    catch {
      case error: IOException =>
        Files.deleteIfExists(log)
        throw new IllegalStateException(
          s"the drift runner encodes videos with ffmpeg, which is not installed: ${error.getMessage}"
        )
    }
  private val in = new BufferedOutputStream(process.getOutputStream, 1 << 20)
  private val row = new Array[Byte](width * 3)
  private var written = 0
  // An ffmpeg that stops reading says why in its log, not in the pipe's
  // error: that one is kept for `finish`'s report.
  private var refused: Option[IOException] = None

  /** How many frames it has taken. */
  def frames: Int = written

  /** The next frame. One ffmpeg no longer takes is not an error here: `finish`
    * reports why it stopped.
    */
  def write(frame: BufferedImage): Unit = {
    require(
      frame.getWidth == width && frame.getHeight == height,
      s"a frame of ${frame.getWidth} × ${frame.getHeight} in a video of $width × $height"
    )
    if (refused.isEmpty)
      try {
        (0 until height).foreach { y =>
          (0 until width).foreach { x =>
            val rgb = frame.getRGB(x, y)
            row(3 * x) = (rgb >> 16).toByte
            row(3 * x + 1) = (rgb >> 8).toByte
            row(3 * x + 2) = rgb.toByte
          }
          in.write(row)
        }
        written += 1
      } catch { case error: IOException => refused = Some(error) }
  }

  /** Ends the video on the frames written: ffmpeg writes what it holds and the
    * file's index, and its failure is this one's, in its own words. With
    * `cutSound`, a soundtrack is cut where the frames end — what a video
    * stopped early needs, its sound's file being the whole clip's.
    */
  def finish(cutSound: Boolean): Unit = {
    try in.close()
    catch { case error: IOException => refused = refused.orElse(Some(error)) }
    val status = process.waitFor()
    val said = Files.readString(log).trim
    Files.deleteIfExists(log)
    if (status != 0 || refused.isDefined)
      throw new IllegalStateException(
        s"ffmpeg ${refused.fold(s"failed ($status)")(error =>
            s"stopped reading the frames (${error.getMessage}, status $status)"
          )}: $said"
      )
    if (cutSound && sound.isDefined && written > 0) {
      val cut = output.resolveSibling(s"${output.getFileName}.cut")
      val seconds = written * rate.denominator.toDouble / rate.numerator
      try {
        VideoFiles.run(
          Seq("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i") ++
            Seq(output.toString, "-t", f"$seconds%.6f", "-c", "copy") ++
            Seq("-f", "webm", cut.toString)
        )
        Files.move(cut, output, StandardCopyOption.REPLACE_EXISTING)
      } finally Files.deleteIfExists(cut)
    }
  }

  /** Stops the encoder and removes what it wrote: a video nothing came of. */
  def abandon(): Unit = {
    process.destroyForcibly()
    process.waitFor()
    try in.close()
    catch { case _: IOException => () }
    Files.deleteIfExists(log)
    Files.deleteIfExists(output)
    ()
  }
}

/** A video as the file drift saves: a webm (VP8, and Vorbis for a soundtrack),
  * encoded by the `ffmpeg` on the `PATH`. sd-server links libvpx and libwebm
  * itself; the runner leaves both to ffmpeg, a dependency of the runner's video
  * models (`specs/42`, step 14).
  */
object VideoFiles {

  /** A webm written frame by frame into `output` (`WebmWriter`). */
  def writer(
      output: Path,
      width: Int,
      height: Int,
      rate: FrameRate,
      sound: Option[SoundFile]
  ): WebmWriter = new WebmWriter(output, width, height, rate, sound)

  /** A source's sound for the video made of its frames: copied when a webm
    * holds its codec, encoded again (Vorbis, at most stereo) otherwise.
    */
  def soundOf(file: Path, sound: SoundProbe): SoundFile =
    SoundFile(
      file,
      if (Set("vorbis", "opus")(sound.codec)) Seq("-c:a", "copy")
      else Seq("-c:a", "libvorbis", "-ac", math.min(sound.channels, 2).toString)
    )

  /** A generated video, whole. Every frame is kept, whatever the soundtrack
    * lasts: one that runs past the last frame is cut there, a shorter one ends
    * in silence. ffmpeg's own `-shortest` cut the frames instead, and closed
    * its input on a run that still had frames to write — the "Broken pipe" a
    * restored clip failed on once its soundtrack was a second short.
    */
  def webm(video: Video): Array[Byte] = {
    require(video.frames.nonEmpty, "a video without frames")
    val folder = Files.createTempDirectory("drift-video")
    try {
      val output = folder.resolve("video.webm")
      val soundtrack = video.soundtrack.map { track =>
        val wav = folder.resolve("soundtrack.wav")
        val whole = (video.frames.size.toLong * track.rate / video.fps *
          track.channels).min(track.samples.length.toLong).toInt
        Files.write(
          wav,
          wave(track.samples.take(whole), track.channels, track.rate)
        )
        SoundFile(wav, Seq("-c:a", "libvorbis"))
      }
      val encoder = writer(
        output,
        video.frames.head.getWidth,
        video.frames.head.getHeight,
        FrameRate.whole(video.fps),
        soundtrack
      )
      video.frames.foreach(encoder.write)
      encoder.finish(cutSound = false)
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

  /** The rate of frames whose file states none. */
  private val UnstatedRate = FrameRate.whole(24)

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

  /** What `file` holds, by `ffprobe`. */
  def probe(file: Path): MediaProbe = {
    val described = ujson.read(
      run(
        Seq(
          "ffprobe",
          "-v",
          "error",
          "-count_packets",
          "-show_entries",
          "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_packets,nb_frames,channels,sample_rate",
          "-of",
          "json",
          file.toString
        )
      )
    )
    val streams = described("streams").arr.toSeq
    def stream(kind: String) =
      streams.find(_.obj.get("codec_type").exists(_.str == kind))
    MediaProbe(
      stream("video").map { video =>
        def whole(name: String) =
          video.obj.get(name).flatMap(_.strOpt).flatMap(_.toIntOption)
        FramesProbe(
          video("width").num.toInt,
          video("height").num.toInt,
          Seq("avg_frame_rate", "r_frame_rate")
            .flatMap(video.obj.get)
            .flatMap(rate => FrameRate.parse(rate.str))
            .headOption,
          whole("nb_read_packets").orElse(whole("nb_frames")).getOrElse(0)
        )
      },
      stream("audio").map(audio =>
        SoundProbe(
          audio.obj.get("codec_name").fold("")(_.str),
          audio("channels").num.toInt,
          audio("sample_rate").str.toInt
        )
      )
    )
  }

  /** `file`'s frames as `described`, one at a time, brought down to a short
    * edge of `shortEdge` when larger.
    */
  def frames(
      file: Path,
      described: FramesProbe,
      shortEdge: Int = ClipShortEdge
  ): ClipFrames = {
    val scale = math.min(
      1.0,
      shortEdge.toDouble / math.min(described.width, described.height)
    )
    def even(side: Double) = math.max(2, (math.round(side) / 2 * 2).toInt)
    val (width, height) =
      if (scale < 1)
        (even(described.width * scale), even(described.height * scale))
      else (described.width, described.height)
    val log = Files.createTempFile("drift-media", ".log")
    val command =
      Seq("ffmpeg", "-hide_banner", "-loglevel", "error", "-i") ++
        Seq(file.toString, "-an", "-vf") ++
        Seq(s"scale=$width:$height:flags=lanczos") ++
        Seq("-f", "rawvideo", "-pix_fmt", "rgb24", "-")
    val process =
      try new ProcessBuilder(command*).redirectError(log.toFile).start()
      catch {
        case error: IOException =>
          Files.deleteIfExists(log)
          throw new IllegalStateException(
            s"the drift runner decodes media with ffmpeg, which is not installed: ${error.getMessage}"
          )
      }
    new ClipFrames(
      width,
      height,
      described.rate.getOrElse(UnstatedRate),
      described.count,
      process,
      log
    )
  }

  private def decodeFile(file: Path): Media = {
    val described = probe(file)
    val soundtrack = described.sound.map { sound =>
      val channels = math.min(sound.channels, 2)
      val raw = run(
        Seq("ffmpeg", "-hide_banner", "-loglevel", "error", "-i") ++
          Seq(file.toString, "-vn", "-f", "f32le", "-ac", channels.toString) ++
          Seq("-ar", sound.rate.toString, "-")
      )
      val buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
      val samples = new Array[Float](raw.length / 4)
      buffer.asFloatBuffer().get(samples)
      Soundtrack(samples, channels, sound.rate)
    }
    described.frames match {
      case None =>
        Media.Sound(
          soundtrack.getOrElse(
            throw new IllegalArgumentException(
              "a media file with neither frames nor sound"
            )
          )
        )
      case Some(video) =>
        val clip = frames(file, video)
        val decoded =
          try clip.toVector
          finally clip.close()
        if (decoded.isEmpty)
          throw new IllegalArgumentException("a video without frames")
        if (decoded.size == 1 && soundtrack.isEmpty) {
          // A still ffmpeg reads and the JDK doesn't (WebP): decoded again
          // at the stills' larger bound.
          if (math.min(video.width, video.height) <= ClipShortEdge)
            Media.Still(decoded.head)
          else {
            val still = frames(file, video, StillShortEdge)
            try Media.Still(still.next())
            finally still.close()
          }
        } else Media.Clip(decoded, clip.rate.value, soundtrack)
    }
  }

  /** `command`'s standard output; its errors when it fails. */
  private[server] def run(command: Seq[String]): Array[Byte] = {
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
