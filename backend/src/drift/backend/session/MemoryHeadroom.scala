package drift.backend.session

import java.nio.file.{Files, Paths}
import scala.util.Try

/** What memory is really left once a model is loaded
  * (`specs/07-launch-and-supervision.md`): free pages plus the file cache
  * nothing maps. The kernel's own `MemAvailable` counts mapped weight files as
  * reclaimable, which they are only at the price of reading them back from disk
  * on every step — the crawl this exists to warn about.
  */
case class MemoryHeadroom(totalBytes: Long, freeBytes: Long) {

  /** The warning for the sessions `loaded`, when less than `thresholdPercent`
    * of the memory is left.
    */
  def warning(loaded: List[String], thresholdPercent: Int): Option[String] =
    Option.when(freeBytes * 100 < totalBytes * thresholdPercent) {
      val (names, advice) = loaded match {
        case Nil        => ("a model loaded", "")
        case one :: Nil => (s"$one loaded", "")
        case several    =>
          (
            s"${several.init.mkString(", ")} and ${several.last} all loaded",
            " Stop one of them for full speed."
          )
      }
      s"Memory is nearly full with $names: ${MemoryHeadroom.gigabytes(freeBytes)} GB of ${MemoryHeadroom
          .gigabytes(totalBytes)} GB left. Generations may crawl while weights are read back from disk.$advice"
    }
}

object MemoryHeadroom {
  private val Line = """(\w+(?:\(\w+\))?):\s+(\d+) kB""".r

  private def gigabytes(bytes: Long): Long = math.round(bytes / 1e9)

  /** Out of the text of `/proc/meminfo`; `None` when it is not one. */
  def parse(meminfo: String): Option[MemoryHeadroom] = {
    val kilobytes = meminfo.linesIterator.collect { case Line(name, value) =>
      name -> value.toLong
    }.toMap
    for {
      total <- kilobytes.get("MemTotal")
      free <- kilobytes.get("MemFree")
    } yield {
      def of(name: String) = kilobytes.getOrElse(name, 0L)
      val unmappedCache =
        (of("Cached") + of("Buffers") - of("Mapped") - of("Shmem")).max(0L)
      MemoryHeadroom(total * 1024, (free + unmappedCache) * 1024)
    }
  }

  /** This machine's, now; `None` where there is no `/proc/meminfo`. */
  def read(): Option[MemoryHeadroom] =
    Try(Files.readString(Paths.get("/proc/meminfo"))).toOption.flatMap(parse)
}
