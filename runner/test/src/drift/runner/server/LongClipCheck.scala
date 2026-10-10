package drift.runner.server

import java.lang.management.ManagementFactory
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.*

/** A clip past the two gigabytes one array holds, through `VideoUpscale` with
  * an upscaler that restores nothing (`ResizePass`), in a small heap
  * (`bugs/54`): minutes long, so a main and not a test.
  *
  * `java -Xmx512m -cp <runner.test's classpath>
  * drift.runner.server.LongClipCheck [FRAMES] [SCALE] [FOLDER]` — 3000 frames
  * of 640 × 480 (2.8 GB as raw frames) at 30000/1001, ×1 unless said, in a
  * temporary folder unless one is given.
  */
object LongClipCheck {

  def main(arguments: Array[String]): Unit = {
    val frames = arguments.lift(0).fold(3000)(_.toInt)
    val scale = arguments.lift(1).fold(1)(_.toInt)
    val folder = arguments
      .lift(2)
      .fold(Files.createTempDirectory("drift-long-clip"))(Paths.get(_))
    val source = folder.resolve("source.webm")
    val output = folder.resolve("output.webm.part")
    val startedAt = System.nanoTime()
    def seconds = (System.nanoTime() - startedAt) / 1e9
    TestClips.make(
      source,
      "640x480",
      "30000/1001",
      frames * 1001 / 30000.0,
      Some(frames * 1001 / 30000.0 - 0.7),
      TestClips.Webm
    )
    println(f"source of $frames frames made in $seconds%.1f s")
    val upscaleStartedAt = System.nanoTime()
    val made = VideoUpscale.run(
      source,
      output,
      (width, height) => (width * scale, height * scale),
      ResizePass,
      (done, of) =>
        if (of > 0 && done * 10 / of != (done - 1) * 10 / of)
          println(s"  $done/$of"),
      (clip, size, passes) =>
        println(
          s"${clip.count} frames of ${clip.width} × ${clip.height} at ${clip.rate} to ${size._1} × ${size._2}, $passes pass(es): ${clip.count.toLong * clip.width * clip.height * 3 / (1 << 20)} MiB of frames in, ${clip.count.toLong * size._1 * size._2 * 3 / (1 << 20)} MiB out"
        )
    )
    val took = (System.nanoTime() - upscaleStartedAt) / 1e9
    val probed = VideoFiles.probe(output)
    val peak = ManagementFactory.getMemoryPoolMXBeans.asScala
      .filter(_.getType == java.lang.management.MemoryType.HEAP)
      .map(_.getPeakUsage.getUsed)
      .sum
    println(
      f"${made.frames} of ${made.sourceFrames} frames written in $took%.1f s; the file holds ${probed.frames
          .map(_.count)} at ${probed.frames.flatMap(_.rate)}, sound ${probed.sound}; ${Files
          .size(output) / (1 << 20)} MiB; heap peak ${peak / (1 << 20)} MiB of ${Runtime.getRuntime.maxMemory / (1 << 20)}"
    )
    require(made.stopped.isEmpty, s"stopped: ${made.stopped}")
    require(
      probed.frames.exists(_.count == frames),
      "frames missing from the file"
    )
    if (arguments.lift(2).isEmpty) {
      Files.deleteIfExists(source)
      Files.deleteIfExists(output)
      Files.deleteIfExists(folder)
    }
  }
}
