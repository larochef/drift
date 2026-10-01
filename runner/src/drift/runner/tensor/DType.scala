package drift.runner.tensor

import drift.runner.tensor.LittleEndian.*

import java.lang.foreign.MemorySegment

/** How a tensor's elements are stored: a block of `blockElements` values in
  * `blockBytes` bytes (one element per block for the plain types).
  *
  * A storage type is a plug-in (`specs/42`, Weight formats): its layout, its
  * CPU reference decoder here, and later its kernels. Adding one means adding
  * an object and listing it in `DType.all`; nothing matches on types.
  */
abstract class DType(
    val name: String,
    val blockElements: Int,
    val blockBytes: Int,
    /** The type id in a GGUF tensor info, if GGUF has this type. */
    val ggmlId: Option[Int],
    /** The dtype string in a safetensors header, if safetensors has it. */
    val safetensorsName: Option[String]
) {

  def byteSize(elements: Long): Long = {
    require(
      elements % blockElements == 0,
      s"$elements elements are not whole $name blocks of $blockElements"
    )
    elements / blockElements * blockBytes
  }

  /** Decodes the block at `offset` of `source` into `target`, from
    * `targetOffset`. The reference the kernels are checked against: it follows
    * the arithmetic of the format's own reference, operation by operation.
    */
  def decodeBlock(
      source: MemorySegment,
      offset: Long,
      target: Array[Float],
      targetOffset: Int
  ): Unit

  /** Why `bytes`, a run of this type's blocks, cannot be this type — for types
    * whose id could be read by mistake (`RocmFp4`).
    */
  def rejects(bytes: MemorySegment): Option[String] = None

  /** Decodes `elements` values from the start of `source`, onto the heap. */
  final def decode(source: MemorySegment, elements: Int): Array[Float] = {
    val target = new Array[Float](elements)
    val blocks = byteSize(elements) / blockBytes
    var block = 0L
    while (block < blocks) {
      decodeBlock(
        source,
        block * blockBytes,
        target,
        (block * blockElements).toInt
      )
      block += 1
    }
    target
  }

  override def toString: String = name
}

/** A type of one element per block. */
abstract class ElementType(
    name: String,
    bytes: Int,
    ggmlId: Option[Int],
    safetensorsName: Option[String]
) extends DType(name, 1, bytes, ggmlId, safetensorsName) {

  def decodeElement(source: MemorySegment, offset: Long): Float

  def decodeBlock(
      source: MemorySegment,
      offset: Long,
      target: Array[Float],
      targetOffset: Int
  ): Unit = target(targetOffset) = decodeElement(source, offset)
}

object DType {

  object F32 extends ElementType("F32", 4, Some(0), Some("F32")) {
    def decodeElement(source: MemorySegment, offset: Long): Float =
      float32(source, offset)
  }

  object F16 extends ElementType("F16", 2, Some(1), Some("F16")) {
    def decodeElement(source: MemorySegment, offset: Long): Float =
      float16(source, offset)
  }

  /** The top half of an F32. */
  object BF16 extends ElementType("BF16", 2, Some(30), Some("BF16")) {
    def decodeElement(source: MemorySegment, offset: Long): Float =
      java.lang.Float.intBitsToFloat(uint16(source, offset) << 16)
  }

  /** fp8 E4M3 "fn": bias 7, no infinities, NaN at `S.1111.111`. */
  object F8E4M3 extends ElementType("F8_E4M3", 1, None, Some("F8_E4M3")) {
    def decodeElement(source: MemorySegment, offset: Long): Float = {
      val code = uint8(source, offset)
      val exponent = (code >> 3) & 0xf
      val mantissa = code & 0x7
      val magnitude =
        if (exponent == 0xf && mantissa == 0x7) Float.NaN
        else if (exponent == 0) java.lang.Math.scalb(mantissa.toFloat, -9)
        else (8 + mantissa) * java.lang.Math.scalb(1f, exponent - 10)
      if ((code & 0x80) != 0) -magnitude else magnitude
    }
  }

  /** fp8 E5M2: the top byte of an F16, infinities and NaN included. */
  object F8E5M2 extends ElementType("F8_E5M2", 1, None, Some("F8_E5M2")) {
    def decodeElement(source: MemorySegment, offset: Long): Float =
      java.lang.Float.float16ToFloat((uint8(source, offset) << 8).toShort)
  }

  /** Raw int8; a scale tensor, when there is one, is applied by the model. */
  object I8 extends ElementType("I8", 1, Some(24), Some("I8")) {
    def decodeElement(source: MemorySegment, offset: Long): Float =
      int8(source, offset).toFloat
  }

  /** Integers such as positions and token ids; decoded values are exact up to
    * 2²⁴.
    */
  object I32 extends ElementType("I32", 4, Some(26), Some("I32")) {
    def decodeElement(source: MemorySegment, offset: Long): Float =
      int32(source, offset).toFloat
  }

  /** Integers such as a LoRA's `.alpha` scalar; decoded values are exact up to
    * 2²⁴.
    */
  object I64 extends ElementType("I64", 8, Some(27), Some("I64")) {
    def decodeElement(source: MemorySegment, offset: Long): Float =
      int64(source, offset).toFloat
  }

  object U8 extends ElementType("U8", 1, None, Some("U8")) {
    def decodeElement(source: MemorySegment, offset: Long): Float =
      uint8(source, offset).toFloat
  }

  val all: Seq[DType] = Seq(
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I8,
    I32,
    I64,
    U8,
    GgmlQuants.Q4_0,
    GgmlQuants.Q4_1,
    GgmlQuants.Q5_0,
    GgmlQuants.Q5_1,
    GgmlQuants.Q8_0,
    KQuants.Q2_K,
    KQuants.Q3_K,
    KQuants.Q4_K,
    KQuants.Q5_K,
    KQuants.Q6_K,
    IQuants.IQ4_NL,
    IQuants.IQ4_XS,
    RocmFp4.Dual,
    RocmFp4.Fast
  )

  private val byGgml = all.flatMap(t => t.ggmlId.map(_ -> t)).toMap
  private val bySafetensors =
    all.flatMap(t => t.safetensorsName.map(_ -> t)).toMap +
      ("F8_E4M3FN" -> F8E4M3)

  def fromGgmlId(id: Int): Either[String, DType] =
    byGgml
      .get(id)
      .toRight(
        GgmlTypeNames
          .get(id)
          .fold(s"unknown GGML type id $id")(name =>
            s"GGML type $name is not supported yet"
          )
      )

  def fromSafetensorsName(name: String): Either[String, DType] =
    bySafetensors
      .get(name)
      .toRight(s"safetensors dtype $name is not supported")

  /** GGML's type ids the runner does not decode, named for the error. */
  private val GgmlTypeNames: Map[Int, String] = Map(
    9 -> "Q8_1",
    15 -> "Q8_K",
    16 -> "IQ2_XXS",
    17 -> "IQ2_XS",
    18 -> "IQ3_XXS",
    19 -> "IQ1_S",
    20 -> "IQ4_NL",
    21 -> "IQ3_S",
    22 -> "IQ2_S",
    23 -> "IQ4_XS",
    25 -> "I16",
    28 -> "F64",
    29 -> "IQ1_M",
    34 -> "TQ1_0",
    35 -> "TQ2_0",
    39 -> "MXFP4"
  )
}
