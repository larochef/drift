package drift.runner.server

import drift.runner.diffusion.{Images, Media, SeedVr2Options, SeedVr2Pipeline}

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}

import utest.*

/** An upscaler that restores nothing: each frame resized, in SeedVR2's batches
  * (`SeedVr2Pipeline.batched`), one step a batch. What `VideoUpscale` is tested
  * on, the model being the GPU's.
  */
object ResizePass extends UpscalePass {

  private val options = SeedVr2Options.Default

  def steps(frames: Int, width: Int, height: Int): Int =
    if (frames <= options.batch) 1
    else {
      val step = options.batch - options.overlap
      (frames - options.overlap + step - 1) / step
    }

  def upscaled(
      frames: Iterator[BufferedImage],
      count: Int,
      width: Int,
      height: Int,
      tick: () => Unit
  ): Iterator[BufferedImage] =
    SeedVr2Pipeline
      .batched(
        frames
          .map(frame => Images.pixels(Images.resized(frame, width, height))),
        options
      ) { (batch, _) =>
        tick()
        batch
      }
      .map(Images.toImage(_, width, height))
}

/** Test clips out of `ffmpeg`'s own sources. */
object TestClips {

  /** A clip of `seconds` of `size` frames at `rate`, with `sound` seconds of a
    * tone if any, encoded as `codecs` say into `file`.
    */
  def make(
      file: Path,
      size: String,
      rate: String,
      seconds: Double,
      sound: Option[Double],
      codecs: Seq[String]
  ): Path = {
    val command =
      Seq("ffmpeg", "-hide_banner", "-loglevel", "error", "-y") ++
        Seq("-f", "lavfi", "-i") ++
        Seq(s"testsrc=size=$size:rate=$rate:duration=$seconds") ++
        sound.toSeq.flatMap(lasting =>
          Seq("-f", "lavfi", "-i", s"sine=frequency=440:duration=$lasting")
        ) ++ codecs ++ Seq(file.toString)
    val status = new ProcessBuilder(command*).inheritIO().start().waitFor()
    require(status == 0, s"ffmpeg could not make $file ($status)")
    file
  }

  val Webm: Seq[String] =
    Seq("-c:v", "libvpx", "-b:v", "2M", "-c:a", "libvorbis")
  val Mp4: Seq[String] = Seq("-c:v", "mpeg4", "-q:v", "4", "-c:a", "aac")
}

/** A video upscaled from a file into a file (`bugs/54`): every frame kept at
  * the source's own rate, whatever its soundtrack lasts, and the frames made
  * kept when the run stops. Needs the `ffmpeg` the runner encodes with.
  */
object VideoUpscaleTests extends TestSuite {

  private def within[A](run: Path => A): A = {
    val folder = Files.createTempDirectory("drift-upscale-test")
    try run(folder)
    finally
      Files
        .walk(folder)
        .sorted(java.util.Comparator.reverseOrder())
        .forEach(path => Files.deleteIfExists(path))
  }

  private def upscale(
      source: Path,
      output: Path,
      scale: Int,
      progress: (Int, Int) => Unit = (_, _) => ()
  ) =
    VideoUpscale.run(
      source,
      output,
      (width, height) => (width * scale, height * scale),
      ResizePass,
      progress,
      (_, _, _) => ()
    )

  private def soundSeconds(file: Path) =
    VideoFiles.decode(Files.readAllBytes(file)) match {
      case Media.Clip(_, _, Some(track)) =>
        track.samples.length.toDouble / track.channels / track.rate
      case other => throw new IllegalStateException(s"no soundtrack: $other")
    }

  val tests = Tests {

    test("every frame at the source's own rate") {
      within { folder =>
        val source = TestClips.make(
          folder.resolve("source.webm"),
          "320x240",
          "30000/1001",
          3.0,
          Some(1.0),
          TestClips.Webm
        )
        val described = VideoFiles.probe(source).frames.get
        assert(described.count == 90)
        assert(described.rate.contains(FrameRate(30000, 1001)))
        val output = folder.resolve("output.webm.part")
        val made = upscale(source, output, 2)
        assert(made.frames == 90 && made.sourceFrames == 90)
        assert(made.stopped.isEmpty)
        assert(made.rate == FrameRate(30000, 1001))
        val probed = VideoFiles.probe(output)
        assert(probed.frames.get.count == 90)
        assert(probed.frames.get.rate.contains(FrameRate(30000, 1001)))
        assert(
          (probed.frames.get.width, probed.frames.get.height) == (640, 480)
        )
        // a soundtrack shorter than the frames: kept, and no frame cut for it
        assert(probed.sound.exists(_.codec == "vorbis"))
      }
    }

    test("a soundtrack longer than the frames keeps every frame") {
      within { folder =>
        val source = TestClips.make(
          folder.resolve("source.mp4"),
          "320x240",
          "25",
          1.2,
          Some(4.0),
          TestClips.Mp4
        )
        val output = folder.resolve("output.webm")
        val made = upscale(source, output, 2)
        assert(made.frames == 30)
        val probed = VideoFiles.probe(output)
        assert(probed.frames.get.count == 30)
        assert(probed.sound.isDefined)
      }
    }

    test("two passes feed one another") {
      within { folder =>
        val source = TestClips.make(
          folder.resolve("source.webm"),
          "160x120",
          "24",
          2.0,
          None,
          TestClips.Webm
        )
        val output = folder.resolve("output.webm")
        var last = (0, 0)
        val made = upscale(source, output, 4, (done, of) => last = (done, of))
        assert(made.frames == 48)
        // 48 frames are 3 batches a pass, and the bar is both passes', the
        // second counting four times the first (four times the pixels)
        assert(last == (15, 15))
        val probed = VideoFiles.probe(output).frames.get
        assert(probed.count == 48)
        assert((probed.width, probed.height) == (640, 480))
      }
    }

    test("a run stopped keeps the video of the frames made") {
      within { folder =>
        val source = TestClips.make(
          folder.resolve("source.webm"),
          "320x240",
          "30",
          3.0,
          Some(3.0),
          TestClips.Webm
        )
        val output = folder.resolve("output.webm")
        val stop = new RuntimeException("stopped")
        val made = upscale(
          source,
          output,
          2,
          (done, _) => if (done == 2) throw stop
        )
        // the first batch's frames, less those the second would blend
        assert(made.frames == 17 && made.sourceFrames == 90)
        assert(made.stopped.contains(stop))
        assert(VideoFiles.probe(output).frames.get.count == 17)
        val sound = soundSeconds(output)
        assert(sound < 17 / 30.0 + 0.1)
      }
    }

    test("a run stopped before its first frame leaves no video") {
      within { folder =>
        val source = TestClips.make(
          folder.resolve("source.webm"),
          "320x240",
          "30",
          1.0,
          None,
          TestClips.Webm
        )
        val output = folder.resolve("output.webm")
        val stop = new RuntimeException("stopped")
        val thrown = assertThrows[RuntimeException](
          upscale(source, output, 2, (_, _) => throw stop)
        )
        assert(thrown eq stop)
        assert(!Files.exists(output))
      }
    }
  }
}
