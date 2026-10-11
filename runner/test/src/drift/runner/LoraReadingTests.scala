package drift.runner

import drift.runner.lora.Lora
import drift.runner.ops.CpuOps

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.nio.{ByteBuffer, ByteOrder}

import utest.*

/** What `Lora.open` reads from the forms published files take, on files written
  * here: where the scale comes from, and what is a pair, a whole weight or
  * neither.
  */
object LoraReadingTests extends TestSuite {

  /** A safetensors file of F32 tensors (name, shape, values) with `metadata`.
    */
  private def written(
      tensors: Seq[(String, Seq[Int], Array[Float])],
      metadata: Map[String, String]
  ): Path = {
    val offsets = tensors.scanLeft(0)((at, tensor) => at + 4 * tensor._3.length)
    val header = ujson.Obj.from(
      Seq(
        "__metadata__" -> (ujson.Obj
          .from(metadata.view.mapValues(ujson.Str(_))): ujson.Value)
      ) ++
        tensors.zip(offsets).map { case ((name, shape, values), at) =>
          name -> (ujson.Obj(
            "dtype" -> "F32",
            "shape" -> ujson.Arr.from(shape),
            "data_offsets" -> ujson.Arr(at, at + 4 * values.length)
          ): ujson.Value)
        }
    )
    // padded as the writers do, so that the data starts on eight bytes
    val text = ujson.write(header)
    val json =
      (text + " " * ((8 - text
        .getBytes(StandardCharsets.UTF_8)
        .length % 8) % 8))
        .getBytes(StandardCharsets.UTF_8)
    val buffer = ByteBuffer
      .allocate(8 + json.length + offsets.last)
      .order(ByteOrder.LITTLE_ENDIAN)
    buffer.putLong(json.length.toLong).put(json)
    tensors.foreach(_._3.foreach(buffer.putFloat))
    val file = Files.createTempFile("drift-lora", ".safetensors")
    Files.write(file, buffer.array())
    file
  }

  private def pair(
      stem: String,
      down: String,
      up: String,
      rank: Int
  ): Seq[(String, Seq[Int], Array[Float])] = Seq(
    (stem + down, Seq(rank, 8), Array.fill(rank * 8)(0.5f)),
    (stem + up, Seq(4, rank), Array.fill(4 * rank)(0.25f))
  )

  private def reading[A](
      tensors: Seq[(String, Seq[Int], Array[Float])],
      metadata: Map[String, String] = Map.empty
  )(check: Lora => A): A = {
    val file = written(tensors, metadata)
    val ops = new CpuOps
    try {
      val lora = Lora.open(ops, file)
      try check(lora)
      finally lora.close()
    } finally {
      ops.close()
      Files.delete(file)
    }
  }

  val tests = Tests {
    test("a pair without alpha has the scale of PEFT's saved settings") {
      val settings = ujson.write(
        ujson.Obj(
          "transformer.lora_alpha" -> 4,
          "transformer.r" -> 2,
          "transformer.use_rslora" -> false,
          "transformer.alpha_pattern" -> ujson.Obj("blocks.1.to_q" -> 1)
        )
      )
      reading(
        pair(
          "transformer.blocks.0.to_q",
          ".lora_A.weight",
          ".lora_B.weight",
          2
        ) ++
          pair(
            "transformer.blocks.1.to_q",
            ".lora_A.weight",
            ".lora_B.weight",
            2
          ),
        Map("format" -> "pt", "lora_adapter_metadata" -> settings)
      ) { lora =>
        assert(lora.pairs("blocks.0.to_q").scale == 2f)
        assert(lora.pairs("blocks.1.to_q").scale == 0.5f)
        assert(lora.unread.isEmpty && lora.whole.isEmpty)
      }
    }
    test("an alpha tensor comes before the settings and none means one") {
      reading(
        pair("a", ".lora_down.weight", ".lora_up.weight", 2) ++
          Seq(("a.alpha", Seq.empty[Int], Array(1f))) ++
          pair("b", ".lora_down.weight", ".lora_up.weight", 2),
        Map(
          "lora_adapter_metadata" -> """{"transformer.lora_alpha": 8}"""
        )
      ) { lora =>
        assert(lora.pairs("a").scale == 0.5f)
        assert(lora.pairs("b").scale == 4f)
      }
      reading(pair("a", ".lora_A.weight", ".lora_B.weight", 2)) { lora =>
        assert(lora.pairs("a").scale == 1f)
      }
    }
    test("rank-stabilized settings divide by the root of the rank") {
      reading(
        pair("a", ".lora_A.weight", ".lora_B.weight", 4),
        Map(
          "lora_adapter_metadata" ->
            """{"transformer.lora_alpha": 8, "transformer.use_rslora": true}"""
        )
      )(lora => assert(lora.pairs("a").scale == 4f))
    }
    test("pairs named without .weight and whole weights beside them") {
      reading(
        pair("blocks.0.to_q", ".lora_down", ".lora_up", 2) ++ Seq(
          ("proj_out.weight", Seq(3, 4, 8), Array.tabulate(96)(_.toFloat)),
          ("blocks.0.norm_q.weight", Seq(4), Array(1f, 2f, 3f, 4f)),
          ("stray", Seq(2), Array(0f, 0f))
        )
      ) { lora =>
        assert(lora.pairs.keySet == Set("blocks.0.to_q"))
        assert(
          lora.whole.keySet == Set("proj_out.weight", "blocks.0.norm_q.weight")
        )
        assert(lora.stepWeights.keySet == Set("proj_out"))
        assert(
          lora
            .wholeValues("blocks.0.norm_q.weight")
            .sameElements(Array(1f, 2f, 3f, 4f))
        )
        assert(lora.unread == Seq("stray"))
      }
    }
  }
}
