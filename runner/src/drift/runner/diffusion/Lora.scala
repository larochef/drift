package drift.runner.diffusion

import drift.runner.models.WeightSource
import drift.runner.ops.Ops
import drift.runner.tensor.*

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.nio.file.Path

/** One low-rank update of a weight: `W += scale × up · down`, `down` `[rank,
  * in]` and `up` `[out, rank]` as the file stores them.
  */
final case class LoraPair(down: Tensor, up: Tensor, scale: Float) {
  def rank: Int = down.shape.dimensions.head.toInt
}

/** A LoRA file's updates by target (the weight's name without `.weight` and
  * without the file's prefix, as the file names it; each model maps the namings
  * it reads), at multiplier 1: `scale = alpha / rank` when the file has an
  * `.alpha`; else from PEFT's settings when diffusers saved them in the file's
  * metadata (`lora_adapter_metadata`: `lora_alpha`, `alpha_pattern`,
  * `use_rslora` — a file trained at alpha 128 over rank 64 is twice its
  * matrices, and at 1 it is half a LoRA); else 1 (sd-cpp's and ComfyUI's rule;
  * `ss_network_alpha` is not read by them either). Its tensors stay mapped.
  *
  * Pairs are read in the forms the published files use: `lora_A` / `lora_B` or
  * `lora_down` / `lora_up` (with or without `.weight`), under
  * `diffusion_model.`, `model.diffusion_model.` (which `WeightNames` drops at
  * load) or diffusers' `transformer.`. Updates not stored as BF16 are converted
  * to it (a rank narrower than the dense kernels' blocks then runs as a BF16
  * GEMM).
  *
  * A file may also hold whole weights beside its pairs (alibaba-pai's
  * step-distilled exports): `whole`, by the weight's own name, for the model
  * that knows what to make of them — a `[steps, out, in]` stack is one output
  * layer per sampling step (`stepWeights`), anything else replaces the model's
  * weight of that name.
  */
final class Lora private (
    ops: Ops,
    source: WeightSource,
    val pairs: Map[String, LoraPair],
    val whole: Map[String, Tensor],
    storedWhole: Map[String, String],
    val unread: Seq[String],
    converted: Seq[Tensor]
) extends AutoCloseable {

  /** The whole weights stacked by sampling step, by target. */
  def stepWeights: Map[String, Tensor] =
    whole.collect {
      case (name, tensor)
          if name.endsWith(".weight") && tensor.shape.dimensions.size == 3 =>
        name.stripSuffix(".weight") -> tensor
    }

  /** A whole weight's values, on the host. */
  def wholeValues(name: String): Array[Float] =
    source.hostValues(storedWhole(name))

  /** A tensor of the file that is no pair (one of `unread`), as F32 values:
    * ComfyUI's whole-tensor differences (`.diff` of a weight, `.diff_b` of a
    * bias), for the models that take them.
    */
  def values(name: String): Array[Float] = {
    val tensor = source(name)
    if (tensor.dtype == DType.F32) ops.toFloats(tensor)
    else {
      val wide = ops.allocate(DType.F32, tensor.shape)
      try {
        ops.convert(tensor, wide)
        ops.toFloats(wide)
      } finally ops.release(wide)
    }
  }

  def close(): Unit = {
    converted.foreach(ops.release)
    source.close()
  }
}

object Lora {

  /** diffusers' prefix; ComfyUI's are gone by the time the names are read. */
  private val DiffusersPrefix = "transformer."
  private val Down = Seq(
    ".lora_A.weight",
    ".lora_down.weight",
    ".lora_A.default.weight",
    ".lora_down"
  )
  private val Up = Seq(
    ".lora_B.weight",
    ".lora_up.weight",
    ".lora_B.default.weight",
    ".lora_up"
  )

  /** PEFT's settings as diffusers saves them with a LoRA (`save_lora_weights`:
    * the safetensors metadata `lora_adapter_metadata`, each key under its
    * component): the scale of a target's update of `rank`, when they give an
    * alpha.
    */
  final private class AdapterSettings(settings: Map[String, ujson.Value]) {
    private def setting(name: String) =
      settings.get(DiffusersPrefix + name).orElse(settings.get(name))
    private val alpha = setting("lora_alpha").flatMap(_.numOpt)
    private val alphaByModule = setting("alpha_pattern")
      .flatMap(_.objOpt)
      .fold(Map.empty[String, Double])(
        _.flatMap((module, value) => value.numOpt.map(module -> _)).toMap
      )
    private val rankStabilized =
      setting("use_rslora").flatMap(_.boolOpt).getOrElse(false)

    def scale(target: String, rank: Long): Option[Float] =
      alphaByModule
        .collectFirst {
          case (module, value)
              if target == module || target.endsWith(s".$module") =>
            value
        }
        .orElse(alpha)
        .map(value =>
          (value / (if (rankStabilized) math.sqrt(rank.toDouble)
                    else rank.toDouble)).toFloat
        )
  }

  private def adapterSettings(source: WeightSource): AdapterSettings =
    new AdapterSettings(
      source.metadata
        .get("lora_adapter_metadata")
        .flatMap(text => scala.util.Try(ujson.read(text).obj.toMap).toOption)
        .getOrElse(Map.empty)
    )

  def open(ops: Ops, path: Path): Lora = {
    val source = WeightSource.open(ops, path)
    try {
      def strip(name: String) =
        name.stripPrefix(DiffusersPrefix)
      val downs = source.names.flatMap(name =>
        Down.find(name.endsWith).map(s => strip(name.stripSuffix(s)) -> name)
      )
      val settings = adapterSettings(source)
      val converted = scala.collection.mutable.ArrayBuffer.empty[Tensor]
      def bf16(tensor: Tensor): Tensor =
        if (tensor.dtype == DType.BF16) tensor
        else {
          val copy = ops.allocate(DType.BF16, tensor.shape)
          if (tensor.dtype == DType.F32) ops.convert(tensor, copy)
          else {
            // F16 (and the rest) by way of F32: no backend converts directly
            val wide = ops.allocate(DType.F32, tensor.shape)
            try {
              ops.convert(tensor, wide)
              ops.convert(wide, copy)
            } finally ops.release(wide)
          }
          converted += copy
          copy
        }
      val pairs = downs.map { (target, downName) =>
        val suffix = Down.find(downName.endsWith).get
        val stem = downName.stripSuffix(suffix)
        val upName = Up
          .map(stem + _)
          .find(source.has)
          .getOrElse(
            throw new IllegalArgumentException(
              s"$path: $downName has no up matrix"
            )
          )
        val down = source(downName)
        val rank = down.shape.dimensions.head
        val scale =
          Some(stem + ".alpha")
            .filter(source.has)
            .map(scalar(source, _) / rank)
            .orElse(settings.scale(target, rank))
            .getOrElse(1f)
        target -> LoraPair(bf16(down), bf16(source(upName)), scale)
      }.toMap
      val read = downs.flatMap { (_, downName) =>
        val stem = downName.stripSuffix(Down.find(downName.endsWith).get)
        Seq(downName, stem + ".alpha") ++ Up.map(stem + _)
      }.toSet
      val storedWhole = source.names
        .filterNot(read)
        .filter(name => name.endsWith(".weight") || name.endsWith(".bias"))
        .map(name => strip(name) -> name)
        .toMap
      new Lora(
        ops,
        source,
        pairs,
        storedWhole.map((name, stored) =>
          name -> (if (source.shape(stored).dimensions.size > 1)
                     bf16(source(stored))
                   else source(stored))
        ),
        storedWhole,
        source.names
          .filterNot(read)
          .filterNot(storedWhole.values.toSet)
          .toSeq
          .sorted,
        converted.toSeq
      )
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }

  private def scalar(source: WeightSource, name: String): Float = {
    val tensor = source(name)
    val segment = tensor.storage match {
      case Storage.Host(s)          => s
      case Storage.Registered(h, _) => h
      case Storage.Device(_, _)     =>
        throw new IllegalStateException(s"$name is not on the host")
    }
    tensor.dtype
      .decode(
        MemorySegment.ofArray(
          segment.asSlice(tensor.byteOffset, tensor.byteSize).toArray(JAVA_BYTE)
        ),
        1
      )
      .head
  }
}

/** The active LoRAs' updates of a model's linear layers, by the model's own
  * name for each (a fused weight's parts each their own), applied at run time:
  * `out += Σ scale × (x · downᵀ) · upᵀ`. The weights stay untouched.
  */
final class LoraUpdates(ops: Ops) extends AutoCloseable {

  private var active = Map.empty[String, Seq[LoraPair]]

  /** Replaces the active updates. */
  def use(updates: Map[String, Seq[LoraPair]]): Unit = active = updates

  /** Scratch for the updates' two products, grown when too small. */
  private var scratch = Option.empty[(Tensor, Tensor)]

  private def grown(low: Long, high: Long): (Tensor, Tensor) =
    scratch
      .filter((l, h) =>
        l.shape.elementCount >= low && h.shape.elementCount >= high
      )
      .getOrElse {
        val sizes = (
          math.max(low, scratch.fold(0L)(_._1.shape.elementCount)),
          math.max(high, scratch.fold(0L)(_._2.shape.elementCount))
        )
        close()
        val made = (
          ops.allocate(DType.F32, Shape.of(sizes._1)),
          ops.allocate(DType.F32, Shape.of(sizes._2))
        )
        scratch = Some(made)
        made
      }

  /** `out += Σ scale × (x · downᵀ) · upᵀ` over the active updates of `site`. */
  def apply(x: Tensor, site: String, out: Tensor): Unit =
    active.get(site).foreach { pairs =>
      val rows = x.shape.dimensions.head
      val columns = out.shape.last
      val (low, high) = grown(rows * pairs.map(_.rank).max, rows * columns)
      pairs.foreach { pair =>
        val reduced = low.prefix(rows, pair.rank.toLong)
        val update = high.prefix(rows, columns)
        ops.linear(x, pair.down, reduced)
        ops.linear(reduced, pair.up, update)
        ops.scale(update, pair.scale, update)
        ops.add(out, update, out)
      }
    }

  def close(): Unit = {
    scratch.foreach((l, h) => { ops.release(l); ops.release(h) })
    scratch = None
  }
}

/** A pipeline's LoRA files, opened on first use and kept mapped for the next
  * requests.
  */
final class LoraFiles(ops: Ops) extends AutoCloseable {

  private val opened = scala.collection.mutable.Map.empty[Path, Lora]

  /** `loras` opened, each with its multiplier, and the problems to report:
    * tensors that are not LoRA pairs, as `file: name` — the whole weights among
    * them too, unless the model answers for those itself (`wholeWeights`).
    */
  def open(
      loras: Seq[(Path, Float)],
      wholeWeights: Boolean = false
  ): (Seq[(Lora, Float)], Seq[String]) = {
    val files = loras.map((path, multiplier) =>
      (path, opened.getOrElseUpdate(path, Lora.open(ops, path)), multiplier)
    )
    (
      files.map((_, lora, multiplier) => lora -> multiplier),
      files.flatMap((path, lora, _) =>
        (lora.unread ++ lora.whole.keys.toSeq.sorted.filterNot(_ =>
          wholeWeights
        )).map(name => s"${path.getFileName}: $name (not a LoRA pair)")
      )
    )
  }

  def close(): Unit = opened.values.foreach(_.close())
}
