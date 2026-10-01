package drift.runner

import utest.*

import drift.runner.diffusion.Lora
import drift.runner.models.Ltx2
import drift.runner.ops.CpuOps

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*

/** François's LTX LoRAs on the official LTX 2.5 transformer (mapped, its
  * linears' weights not loaded): every pair's target one of the model's
  * linears, of the update's shape. The LTX 2.5 files must map fully; the LTX
  * 2.3 ones are reported. Skipped where the files are absent.
  */
object LtxLoraFilesTests extends TestSuite {

  private val home = Paths.get(System.getProperty("user.home"))
  private val transformer: Option[Path] = {
    val snapshots =
      home.resolve(
        ".cache/huggingface/hub/models--Lightricks--LTX-2.5/snapshots"
      )
    Option
      .when(Files.isDirectory(snapshots))(
        Files.list(snapshots).iterator().asScala.toSeq
      )
      .toSeq
      .flatten
      .map(
        _.resolve(
          "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
        )
      )
      .find(Files.exists(_))
  }

  private def loras(version: String): Seq[Path] = {
    val folder = home.resolve(s".cache/drift/loras/$version")
    if (!Files.isDirectory(folder)) Nil
    else
      Files
        .walk(folder)
        .iterator()
        .asScala
        .filter(_.toString.endsWith(".safetensors"))
        .toSeq
        .sorted
  }

  /** Per file: its pairs, the tensors that are not pairs, and the targets the
    * transformer's `useLoras` leaves unapplied.
    */
  private def check(
      files: Seq[Path]
  ): Seq[(Path, Int, Seq[String], Seq[String])] =
    transformer.toSeq.flatMap { path =>
      val ops = new CpuOps
      val model = Ltx2.open(ops, path)
      try
        files.map { file =>
          val lora = Lora.open(ops, file)
          try {
            val unmatched = model.useLoras(Seq(lora -> 1f))
            model.useLoras(Nil)
            println(
              s"  ${file.getFileName}: ${lora.pairs.size} pairs, ${lora.unread.size} not pairs, ${unmatched.size} unmatched"
            )
            (file, lora.pairs.size, lora.unread, unmatched.sorted)
          } finally lora.close()
        }
      finally {
        model.close()
        ops.close()
      }
    }

  val tests = Tests {
    test("LTX 2.5 LoRAs map fully onto LTX 2.5") {
      check(loras("ltx-2.5")).foreach { (_, pairs, unread, unmatched) =>
        assert(pairs > 0, unread.isEmpty, unmatched.isEmpty)
      }
    }
    test("LTX 2.3 LoRAs against LTX 2.5's shapes (reported)") {
      val results = check(loras("ltx-2.3"))
      val full = results.count((_, _, unread, unmatched) =>
        unread.isEmpty && unmatched.isEmpty
      )
      println(s"  $full of ${results.size} would apply fully")
    }
  }
}
