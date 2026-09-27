package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Model conversion (`specs/25-model-conversion.md`): a cached weight file
  * turned into a GGUF quant by sd-cpp's own converter (`sd-cli -M convert`),
  * written under the family's directory of the drift cache and registered as a
  * model of that family.
  */

/** ComfyUI's quantization marker on a safetensors file (`<module>.comfy_quant`
  * tensors holding JSON). sd-cpp never requantizes such tensors, so a file
  * carrying one cannot be converted until drift dequantizes it first.
  */
case class ComfyQuantInfo(format: String, convrot: Boolean, groupSize: Int)

/** What a weight file holds, read from its header alone: how many tensors and
  * parameters, and how many bytes of each storage type — the dtype mix that
  * says whether a file is bf16, fp8 or int8 before anything is converted.
  */
case class ModelFileInfo(
    /** "safetensors" or "gguf". */
    format: String,
    tensorCount: Int,
    parameterCount: Long,
    /** Bytes of tensor data per storage type, sd-cpp's lowercase type names
      * (`bf16`, `f16`, `f8_e4m3`, `i8`, `q8_0`, …).
      */
    bytesByType: Map[String, Long],
    comfyQuant: Option[ComfyQuantInfo] = None,
    /** fp8 weights with scale tensors — ComfyUI's `scaled_fp8` marker with
      * `.scale_weight`, or Ideogram's `.weight_scale`. sd-cpp's converter does
      * not apply the scales (upstream rescales to bf16 with a script first), so
      * drift dequantizes such a file before converting, like an int8 one.
      */
    scaledFp8: Boolean = false
)
object ModelFileInfo {
  given JsonValueCodec[ModelFileInfo] = JsonCodecMaker.make
}

/** The target types the modal offers, with the bits a weight takes in each so
  * an output size can be estimated from the parameter count. sd-cpp keeps
  * biases, norms, embeddings and the in/out projections unquantized, so the
  * real file comes out a little larger than the estimate.
  */
case class ConversionType(name: String, label: String, bitsPerWeight: Double)

object ConversionTypes {
  val all: List[ConversionType] = List(
    ConversionType("q8_0", "Q8_0 — near lossless", 8.5),
    ConversionType("q6_K", "Q6_K", 6.56),
    ConversionType("q5_K", "Q5_K", 5.5),
    ConversionType("q4_K", "Q4_K — the usual choice", 4.5),
    ConversionType("iq4_xs", "IQ4_XS — smaller, needs a good source", 4.25),
    ConversionType("q3_K", "Q3_K — visible loss", 3.44),
    ConversionType("f16", "F16 — no quantization", 16),
    ConversionType("bf16", "BF16 — no quantization", 16)
  )

  val default: String = "q8_0"

  def byName(name: String): Option[ConversionType] = all.find(_.name == name)

  /** The estimated output size of `parameterCount` weights at `typeName`. */
  def estimateBytes(parameterCount: Long, typeName: String): Option[Long] =
    byName(typeName).map(t => (parameterCount * t.bitsPerWeight / 8).toLong)
}

enum ConversionState derives CanEqual {
  case Queued, Dequantizing, Converting, Registering, Completed, Failed,
    Cancelled

  def isActive: Boolean = this match {
    case Queued | Dequantizing | Converting | Registering => true
    case _                                                => false
  }
}
object ConversionState {
  given Schema[ConversionState] =
    Schema.derivedEnumeration[ConversionState].defaultStringBased
}

/** What the Model Cache page asks for: one cached file, a family, a type. */
case class ConversionRequest(
    /** The cached file's real path, as the inventory lists it. */
    path: String,
    familyId: String,
    /** A sd-cpp type name, spelled as sd-cpp spells it (`q4_K`). */
    targetType: String,
    /** Raw `--tensor-type-rules`, empty for a uniform type. */
    rules: String = "",
    /** The output file name, `.gguf` included. */
    outputName: String,
    threads: Option[Int] = None,
    /** For an int8 or scaled fp8 source: keep the dequantized safetensors drift
      * writes before converting, instead of removing it afterwards — the file
      * to run on the GPU when judging the dequantization itself.
      */
    keepIntermediate: Boolean = false
)
object ConversionRequest {
  given JsonValueCodec[ConversionRequest] = JsonCodecMaker.make
}

/** One conversion of this drift run. `progress` counts tensors written, as
  * sd-cli's bar reports them; `modelId` is the registered model once done.
  */
case class ConversionJob(
    id: String,
    sourcePath: String,
    sourceLabel: String,
    outputPath: String,
    familyId: String,
    targetType: String,
    rules: String,
    state: ConversionState,
    progress: Option[PostProcessProgress] = None,
    /** sd-cli's own tail of the bar ("1.20GB/s"), or the current step. */
    detail: String = "",
    error: Option[String] = None,
    startedAt: Long,
    completedAt: Option[Long] = None,
    logTail: List[String] = List.empty,
    modelId: Option[String] = None
)
object ConversionJob {
  given JsonValueCodec[ConversionJob] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[ConversionJob]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given Schema[ConversionJob] = Schema.derived
}

// `base` in Api.scala is private to that file (tracked as bug 16); the same
// prefix is rebuilt here so these endpoints live under /api like the rest.
private val conversionBase = endpoint.in("api")

/** What a cached file holds, from its header. The failure is the reason the
  * file could not be read as a weight file.
  */
val inspectCachedFile: PublicEndpoint[String, String, ModelFileInfo, Any] =
  conversionBase.get
    .in("cache" / "files" / "inspect")
    .in(query[String]("path"))
    .errorOut(stringBody)
    .out(jsonBody[ModelFileInfo])

/** Queues a conversion. The failure is why the request was refused — a bad
  * name, an output that already exists, a source sd-cpp cannot convert.
  */
val startConversion
    : PublicEndpoint[ConversionRequest, String, ConversionJob, Any] =
  conversionBase.post
    .in("conversions")
    .in(jsonBody[ConversionRequest])
    .errorOut(stringBody)
    .out(jsonBody[ConversionJob])

/** Every conversion of this drift run, newest first. */
val listConversions: PublicEndpoint[Unit, Unit, List[ConversionJob], Any] =
  conversionBase.get.in("conversions").out(jsonBody[List[ConversionJob]])

/** Cancels a queued or running conversion; its partial output is removed. */
val cancelConversion: PublicEndpoint[String, Unit, Boolean, Any] =
  conversionBase.post
    .in("conversions" / path[String] / "cancel")
    .out(jsonBody[Boolean])
