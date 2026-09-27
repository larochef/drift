package drift.runner.native

import drift.runner.tensor.MappedFile

import java.lang.foreign.MemorySegment
import java.nio.file.Path

/** A file mapped read-only and registered with the GPU: the host and the GPU
  * read the same pages, with no copy, so a model loads at page-cache speed.
  */
final class RegisteredFile(hip: HipRuntime, val path: Path)
    extends AutoCloseable {

  private val mapped = new MappedFile(path)

  val host: MemorySegment = mapped.segment

  val device: MemorySegment =
    hip.register(host, HipRuntime.RegisterMapped | HipRuntime.RegisterReadOnly)

  def close(): Unit = {
    hip.unregister(host)
    mapped.close()
  }
}
