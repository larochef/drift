package drift.runner

import drift.runner.diffusion.Lora
import drift.runner.formats.SafetensorsModel
import drift.runner.models.{Wan, WanConfig, WeightSource}
import drift.runner.ops.CpuOps

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*

/** Maps every LoRA file under a folder onto a Wan transformer's sites, on the
  * reference backend (the files mapped, the F32 ones converted; nothing on the
  * GPU): `[LORA_FOLDER [TRANSFORMER]]`, by default François's Wan 2.2 14B LoRAs
  * and his low-noise expert (only its header is read, for the sizes). Fails
  * unless every tensor of every file is read (no unsupported type, no name
  * outside a pair) and every pair lands on a site of its shape.
  */
object WanLoraCheck {

  private val Cache = Paths.get(System.getProperty("user.home"), ".cache/drift")

  private def files(folder: Path, suffix: String): Seq[Path] =
    Files
      .walk(folder)
      .iterator()
      .asScala
      .filter(_.getFileName.toString.endsWith(suffix))
      .toSeq
      .sorted

  def main(arguments: Array[String]): Unit = {
    val folder =
      arguments.headOption
        .map(Paths.get(_))
        .getOrElse(Cache.resolve("loras/wan-2.2-14B"))
    val transformer = arguments
      .lift(1)
      .map(Paths.get(_))
      .getOrElse(
        files(Cache.resolve("models/wan-2.2-14B-low-noise"), ".gguf").head
      )
    val ops = new CpuOps
    try {
      val source = WeightSource.open(ops, transformer)
      val config =
        try WanConfig.of(source)
        finally source.close()
      println(s"$config (${transformer.getFileName})")
      val site = Wan.loraSite(config)
      val problems = files(folder, ".safetensors").flatMap { path =>
        val header = SafetensorsModel.open(path)
        val unsupported =
          try header.unsupported.map((name, reason) => s"$name: $reason").toSeq
          finally header.close()
        val lora = Lora.open(ops, path)
        try {
          val placed = lora.pairs.toSeq.map((target, pair) =>
            (target, pair, site(target, pair))
          )
          val scales = placed.map(_._2.scale).distinct.sorted
          val ranks = placed.map(_._2.rank).distinct.sorted
          println(
            f"${folder.relativize(path)}: ${placed.size} pairs on ${placed.flatMap(_._3).distinct.size} sites, " +
              s"rank ${ranks.mkString("/")}, scale ${scales.mkString("/")}"
          )
          (unsupported ++ lora.unread.map(name => s"$name: not a LoRA pair") ++
            placed.collect { case (target, _, None) => s"$target: no site" })
            .map(problem => s"${path.getFileName}: $problem")
        } finally lora.close()
      }
      problems.foreach(problem => println(s"  UNREAD $problem"))
      if (problems.nonEmpty)
        throw new AssertionError(
          s"${problems.size} tensors or pairs left unread"
        )
      println("every tensor of every file read, every pair on a site")
    } finally ops.close()
  }
}
