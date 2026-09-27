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
  * `.alpha`, else 1 (sd-cpp's and ComfyUI's rule; metadata such as
  * `ss_network_alpha` is not read by them either). Its tensors stay mapped.
  *
  * Pairs are read in the forms the published files use: `lora_A` / `lora_B` or
  * `lora_down` / `lora_up`, under `diffusion_model.`, `model.diffusion_model.`
  * (which `WeightNames` drops at load) or diffusers' `transformer.`. Updates
  * not stored as BF16 are converted to it (a rank narrower than the dense
  * kernels' blocks then runs as a BF16 GEMM).
  */
final class Lora private (
    ops: Ops,
    source: WeightSource,
    val pairs: Map[String, LoraPair],
    val unread: Seq[String],
    converted: Seq[Tensor]
) extends AutoCloseable {

  def close(): Unit = {
    converted.foreach(ops.release)
    source.close()
  }
}

object Lora {

  /** diffusers' prefix; ComfyUI's are gone by the time the names are read. */
  private val DiffusersPrefix = "transformer."
  private val Down =
    Seq(".lora_A.weight", ".lora_down.weight", ".lora_A.default.weight")
  private val Up =
    Seq(".lora_B.weight", ".lora_up.weight", ".lora_B.default.weight")

  def open(ops: Ops, path: Path): Lora = {
    val source = WeightSource.open(ops, path)
    try {
      def strip(name: String) =
        name.stripPrefix(DiffusersPrefix)
      val downs = source.names.flatMap(name =>
        Down.find(name.endsWith).map(s => strip(name.stripSuffix(s)) -> name)
      )
      val converted = scala.collection.mutable.ArrayBuffer.empty[Tensor]
      def bf16(tensor: Tensor): Tensor =
        if (tensor.dtype == DType.BF16) tensor
        else {
          val copy = ops.allocate(DType.BF16, tensor.shape)
          ops.convert(tensor, copy)
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
        val alpha =
          Some(stem + ".alpha").filter(source.has).map(scalar(source, _))
        target -> LoraPair(
          bf16(down),
          bf16(source(upName)),
          alpha.fold(1f)(_ / down.shape.dimensions.head)
        )
      }.toMap
      val read = downs.flatMap { (_, downName) =>
        val stem = downName.stripSuffix(Down.find(downName.endsWith).get)
        Seq(downName, stem + ".alpha") ++ Up.map(stem + _)
      }.toSet
      new Lora(
        ops,
        source,
        pairs,
        source.names.filterNot(read).toSeq.sorted,
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
    * tensors that are not LoRA pairs, as `file: name`.
    */
  def open(loras: Seq[(Path, Float)]): (Seq[(Lora, Float)], Seq[String]) = {
    val files = loras.map((path, multiplier) =>
      (path, opened.getOrElseUpdate(path, Lora.open(ops, path)), multiplier)
    )
    (
      files.map((_, lora, multiplier) => lora -> multiplier),
      files.flatMap((path, lora, _) =>
        lora.unread.map(name => s"${path.getFileName}: $name (not a LoRA pair)")
      )
    )
  }

  def close(): Unit = opened.values.foreach(_.close())
}
