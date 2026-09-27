package drift.runner.native

import java.lang.foreign.{Arena, MemorySegment}

/** One code object built from `runner/kernels/<name>.hip`, loaded from the
  * resource `kernels/<name>.hsaco`.
  */
final class KernelModule(hip: HipRuntime, val name: String)
    extends AutoCloseable {

  private val arena = Arena.ofShared()

  private val module = {
    val resource = s"/kernels/$name.hsaco"
    val stream = Option(getClass.getResourceAsStream(resource)).getOrElse(
      throw new IllegalStateException(s"kernel module $resource is not built")
    )
    val bytes =
      try stream.readAllBytes()
      finally stream.close()
    val image = arena.allocate(bytes.length.toLong, 64)
    MemorySegment.copy(MemorySegment.ofArray(bytes), 0, image, 0, bytes.length)
    hip.loadModule(image)
  }

  def function(functionName: String): KernelFunction =
    hip.function(module, functionName)

  def close(): Unit = {
    hip.unloadModule(module)
    arena.close()
  }
}
