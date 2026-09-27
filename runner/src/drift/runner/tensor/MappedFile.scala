package drift.runner.tensor

import java.lang.foreign.{Arena, MemorySegment}
import java.nio.channels.FileChannel
import java.nio.channels.FileChannel.MapMode
import java.nio.file.{Path, StandardOpenOption}

/** A whole file mapped read-only: its pages come from the page cache, never
  * from the heap. `RegisteredFile` registers one with the GPU.
  */
final class MappedFile(val path: Path) extends AutoCloseable {

  private val arena = Arena.ofShared()

  val segment: MemorySegment = {
    val channel = FileChannel.open(path, StandardOpenOption.READ)
    try {
      if (channel.size() == 0)
        throw new IllegalArgumentException(s"$path is empty")
      channel.map(MapMode.READ_ONLY, 0, channel.size(), arena)
    } finally channel.close()
  }

  def close(): Unit = arena.close()
}
