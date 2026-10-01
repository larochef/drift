package drift.runner

import drift.runner.diffusion.Lora
import drift.runner.models.{MiniMaxH3, MiniMaxH3Partition, WeightSource}
import drift.runner.ops.CpuOps

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*

/** Places every LoRA under a folder on a MiniMax H3 checkpoint, from their
  * headers alone (nothing is computed): `CHECKPOINT LORA_FOLDER`. Prints the
  * checkpoint's partition (by its name and by its fingerprint), then, per file,
  * how many of its pairs land on a weight and every target left unapplied or
  * tensor left unread.
  */
object MiniMaxH3LoraCheck {

  def main(arguments: Array[String]): Unit = {
    val Array(checkpoint, folder) = arguments
    val ops = new CpuOps
    val source = WeightSource.open(ops, Paths.get(checkpoint))
    val shapes =
      try {
        println(
          s"partition: by name ${MiniMaxH3Partition.named(Paths.get(checkpoint))}, by fingerprint ${MiniMaxH3Partition.fingerprinted(source)}"
        )
        MiniMaxH3.siteShapes(source)
      } finally source.close()
    val files = Files
      .walk(Paths.get(folder))
      .iterator()
      .asScala
      .filter(path => path.toString.endsWith(".safetensors"))
      .toSeq
      .sortBy(_.toString)
    var failures = 0
    files.foreach { (path: Path) =>
      val lora = Lora.open(ops, path)
      try {
        val unplaced = lora.pairs.toSeq
          .filter((target, pair) =>
            MiniMaxH3.sitesOf(target, pair, shapes).isEmpty
          )
          .map(_._1)
          .sorted
        println(
          s"${path.getFileName}: ${lora.pairs.size - unplaced.size} of ${lora.pairs.size} pairs placed"
        )
        unplaced.foreach(target => println(s"  unapplied: $target"))
        lora.unread.foreach(name => println(s"  unread: $name"))
        if (unplaced.nonEmpty || lora.unread.nonEmpty) failures += 1
      } finally lora.close()
    }
    println(s"${files.size} files, $failures with problems")
  }
}
