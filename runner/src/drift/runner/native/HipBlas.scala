package drift.runner.native

import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle
import java.nio.file.Files

/** hipBLAS's `GemmEx`, the prefill path of `Ops.linear`, through FFM like
  * `HipRuntime`. One handle, on the default stream; creating it sets
  * `ROCBLAS_USE_HIPBLASLT` for the whole process.
  *
  * hipBLAS is column-major. The runner is row-major, so `gemm` takes row-major
  * operands and swaps them: `C = A · Bᵀ` row-major is `Cᵀ = B · Aᵀ`
  * column-major.
  */
final class HipBlas(hip: HipRuntime) extends AutoCloseable {

  import HipBlas.*

  private val linker = Linker.nativeLinker()
  private val library = {
    val path = hip.rocmRoot.resolve("lib/libhipblas.so")
    if (!Files.isRegularFile(path))
      throw new IllegalStateException(s"no hipBLAS at $path")
    SymbolLookup.libraryLookup(path, Arena.global())
  }

  private def bind(
      name: String,
      arguments: java.lang.foreign.MemoryLayout*
  ): MethodHandle =
    linker.downcallHandle(
      library
        .find(name)
        .orElseThrow(() =>
          new IllegalStateException(s"$name is not in libhipblas")
        ),
      FunctionDescriptor.of(JAVA_INT, arguments*)
    )

  private val hipblasCreate = bind("hipblasCreate", ADDRESS)
  private val hipblasDestroy = bind("hipblasDestroy", ADDRESS)
  private val hipblasGemmEx = bind(
    "hipblasGemmEx",
    ADDRESS, // handle
    JAVA_INT, // transA
    JAVA_INT, // transB
    JAVA_INT, // m
    JAVA_INT, // n
    JAVA_INT, // k
    ADDRESS, // alpha
    ADDRESS, // A
    JAVA_INT, // aType
    JAVA_INT, // lda
    ADDRESS, // B
    JAVA_INT, // bType
    JAVA_INT, // ldb
    ADDRESS, // beta
    ADDRESS, // C
    JAVA_INT, // cType
    JAVA_INT, // ldc
    JAVA_INT, // computeType
    JAVA_INT // algo
  )

  private def check(call: String, status: Int): Unit =
    if (status != 0)
      throw new IllegalStateException(
        s"$call failed with hipBLAS status $status"
      )

  /** rocBLAS picks hipBLASLt's kernels, about 20% faster on gfx1151, only when
    * this is set as the handle is created. The runner sets it itself, through
    * libc, so that no launcher can forget it.
    */
  locally {
    val setenv = linker.downcallHandle(
      linker.defaultLookup().find("setenv").orElseThrow(),
      FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT)
    )
    val arena = Arena.ofConfined()
    try
      check(
        "setenv(ROCBLAS_USE_HIPBLASLT)",
        (setenv.invokeExact(
          arena.allocateFrom("ROCBLAS_USE_HIPBLASLT"),
          arena.allocateFrom("1"),
          1
        ): Int)
      )
    finally arena.close()
  }

  private val handle: MemorySegment = {
    val arena = Arena.ofConfined()
    try {
      val slot = arena.allocate(ADDRESS)
      check("hipblasCreate", (hipblasCreate.invokeExact(slot): Int))
      slot.get(ADDRESS, 0)
    } finally arena.close()
  }

  /** `c[m, n] = a[m, k] · b[n, k]ᵀ`, all row-major F16, accumulated in F32
    * ("HHS"). On gfx1151 this is the tuned path, about 27 TFLOPS at 512 × 8192
    * · 16384 through hipBLASLt; an F32 output runs untuned kernels at 3 to 6.
    * The price is the output's rounding to F16, and its range: a result beyond
    * ±65504 overflows.
    */
  def gemm(
      a: MemorySegment,
      b: MemorySegment,
      c: MemorySegment,
      m: Int,
      n: Int,
      k: Int
  ): Unit = gemm(a, b, c, m, n, k, RealF16)

  /** `gemm` with `a`, `b` and `c` all of `dataType` (`RealF16` or `RealBF16`).
    */
  def gemm(
      a: MemorySegment,
      b: MemorySegment,
      c: MemorySegment,
      m: Int,
      n: Int,
      k: Int,
      dataType: Int
  ): Unit = gemm(a, b, c, m, n, k, dataType, dataType)

  /** `gemm` with `a` and `b` of `dataType` and `c` of `outputType`: `RealF32`
    * keeps the sums unrounded, on untuned kernels.
    */
  def gemm(
      a: MemorySegment,
      b: MemorySegment,
      c: MemorySegment,
      m: Int,
      n: Int,
      k: Int,
      dataType: Int,
      outputType: Int
  ): Unit = {
    val arena = Arena.ofConfined()
    try {
      val one = arena.allocateFrom(JAVA_FLOAT, 1f)
      val zero = arena.allocateFrom(JAVA_FLOAT, 0f)
      check(
        "hipblasGemmEx",
        (hipblasGemmEx.invokeExact(
          handle,
          OperationTranspose,
          OperationNone,
          n,
          m,
          k,
          one,
          b,
          dataType,
          k,
          a,
          dataType,
          k,
          zero,
          c,
          outputType,
          n,
          Compute32F,
          GemmDefault
        ): Int)
      )
    } finally arena.close()
  }

  def close(): Unit =
    check("hipblasDestroy", (hipblasDestroy.invokeExact(handle): Int))
}

object HipBlas {
  val OperationNone = 111 // HIPBLAS_OP_N
  val OperationTranspose = 112 // HIPBLAS_OP_T
  val RealF32 = 0 // HIP_R_32F
  val RealF16 = 2 // HIP_R_16F
  val RealBF16 = 14 // HIP_R_16BF
  val Compute32F = 2 // HIPBLAS_COMPUTE_32F
  val GemmDefault = 160 // HIPBLAS_GEMM_DEFAULT
}
