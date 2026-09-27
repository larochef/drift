package drift.runner

import utest.*

import drift.runner.ops.{CpuOps, HipOps, MatVecInputs}
import drift.runner.tensor.{Comparison, Tolerance}

/** The flash-attention kernel against the reference, over head sizes, GQA
  * groups and every option. The kernel rounds queries and weights to F16, which
  * moves outputs of about 1 by a few 1e-4.
  */
object AttentionTests extends TestSuite {

  private val base = AttentionCase(
    tokens = 37,
    keyCount = 37,
    qHeads = 8,
    kvHeads = 2,
    dimension = 128,
    causal = true,
    window = None,
    softcap = None,
    sinks = false,
    pageSize = 64,
    shuffledPages = true,
    seed = 11
  )

  private def check(cases: AttentionCase*): Unit = {
    val cpu = new CpuOps
    val hip = new HipOps(Gpu.hip, MatVecInputs.Int8)
    try
      cases.foreach { c =>
        val comparison =
          Comparison.of(c.run(cpu), c.run(hip), Tolerance(4e-3, 0))
        println(
          f"  ${c.tokens}%3d queries, ${c.keyCount}%4d keys, ${c.qHeads}/${c.kvHeads} heads of ${c.dimension}: ${comparison.summary}"
        )
        assert(comparison.passed)
      }
    finally {
      hip.close()
      cpu.close()
    }
  }

  val tests = Tests {
    test("prefill, each head size") {
      check(Seq(64, 128, 256).map(d => base.copy(dimension = d))*)
    }
    test("GQA groups of 1, 4 and 12 (Qwen 3.8: 24 over 2)") {
      check(
        base.copy(qHeads = 4, kvHeads = 4),
        base.copy(qHeads = 8, kvHeads = 2),
        base.copy(qHeads = 24, kvHeads = 2, dimension = 256)
      )
    }
    test("decode and MTP-sized steps over a long cache (split keys)") {
      check(
        base.copy(tokens = 1, keyCount = 1500),
        base.copy(
          tokens = 4,
          keyCount = 777,
          qHeads = 24,
          kvHeads = 2,
          dimension = 256
        ),
        base.copy(
          tokens = 1,
          keyCount = 3000,
          qHeads = 4,
          kvHeads = 4,
          dimension = 64
        )
      )
    }
    test("an encoder: no causal mask") {
      check(
        base.copy(causal = false),
        base.copy(causal = false, tokens = 50, keyCount = 50, dimension = 64)
      )
    }
    test("many rows (the tiled kernel): prefill, chunks, encoders, options") {
      check(
        base.copy(tokens = 100, keyCount = 100),
        base.copy(
          tokens = 90,
          keyCount = 300,
          qHeads = 12,
          kvHeads = 4,
          dimension = 64
        ),
        base.copy(
          tokens = 300,
          keyCount = 300,
          qHeads = 4,
          kvHeads = 4,
          causal = false
        ),
        base.copy(
          tokens = 130,
          keyCount = 400,
          window = Some(64),
          softcap = Some(30f),
          sinks = true
        ),
        base.copy(
          tokens = 70,
          keyCount = 70,
          qHeads = 16,
          kvHeads = 4,
          causal = false,
          pageSize = 16
        )
      )
    }
    test("sliding window, softcap, sinks") {
      check(
        base.copy(tokens = 20, keyCount = 300, window = Some(64)),
        base.copy(softcap = Some(30f)),
        base.copy(sinks = true),
        base.copy(tokens = 1, keyCount = 1500, sinks = true, window = Some(512))
      )
    }
  }
}
