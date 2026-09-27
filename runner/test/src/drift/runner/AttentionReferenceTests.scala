package drift.runner

import utest.*

import drift.runner.ops.{Attention, CpuOps}
import drift.runner.state.{PageAllocator, SequencePages}
import drift.runner.tensor.{Comparison, DType, Shape, Tolerance}

/** The reference attention and the page bookkeeping, against hand-worked values
  * and invariants.
  */
object AttentionReferenceTests extends TestSuite {

  private def withCpu[A](body: CpuOps => A): A = {
    val ops = new CpuOps
    try body(ops)
    finally ops.close()
  }

  private val plain = AttentionCase(
    tokens = 5,
    keyCount = 40,
    qHeads = 4,
    kvHeads = 2,
    dimension = 16,
    causal = true,
    window = None,
    softcap = None,
    sinks = false,
    pageSize = 16,
    shuffledPages = false,
    seed = 7
  )

  val tests = Tests {
    test("pages are handed out, truncated and returned") {
      val allocator = new PageAllocator(4)
      val sequence = new SequencePages(allocator, 16)
      sequence.reserve(33) // three pages
      assert(sequence.table.length == 3 && allocator.available == 1)
      sequence.truncate(17) // positions 0..16: two pages
      assert(sequence.table.length == 2 && allocator.available == 2)
      intercept[IllegalStateException](sequence.reserve(16 * 5))
      sequence.release()
      assert(allocator.available == 4)
    }
    test("one visible key: the output is its value") {
      withCpu { ops =>
        val cache = ops.allocateCache(1, 16, 1, 16)
        val table = ops.fromInts(Shape.of(1), Array(0))
        val value = Array.tabulate(16)(_.toFloat / 8) // exact in F16
        ops.cacheWrite(
          ops.fromFloats(Shape.of(1, 1, 16), Array.fill(16)(0.5f)),
          ops.fromFloats(Shape.of(1, 1, 16), value),
          cache,
          table,
          0
        )
        val out = ops.allocate(DType.F32, Shape.of(1, 1, 16))
        ops.attention(
          ops.fromFloats(Shape.of(1, 1, 16), Array.fill(16)(1f)),
          cache,
          table,
          0,
          1,
          Attention(0.25f, true, None, None, None),
          out
        )
        assert(ops.toFloats(out).sameElements(value))
        // a sink of the same logit as the key (q·k × scale = 2) takes half the weight
        val sinks = ops.fromFloats(Shape.of(1), Array(2f))
        ops.attention(
          ops.fromFloats(Shape.of(1, 1, 16), Array.fill(16)(1f)),
          cache,
          table,
          0,
          1,
          Attention(0.25f, true, None, None, Some(sinks)),
          out
        )
        assert(ops.toFloats(out).sameElements(value.map(_ / 2)))
      }
    }
    test("the order of the pages does not matter") {
      withCpu { ops =>
        val shuffled = plain.copy(shuffledPages = true)
        assert(!shuffled.table.sameElements(plain.table))
        assert(shuffled.run(ops).sameElements(plain.run(ops)))
      }
    }
    test("prefill then decode equals one longer prefill") {
      withCpu { ops =>
        // the last query of a 5-token prefill sees what a 1-token decode at the same position sees
        val prefill = plain.run(ops)
        val decode = plain
          .copy(
            tokens = 1,
            givenQueries = Some(plain.queries.takeRight(4 * 16))
          )
          .run(ops)
        val last = prefill.takeRight(4 * 16)
        assert(Comparison.of(last, decode, Tolerance.Exact).passed)
      }
    }
    test("a window of the whole context changes nothing; a narrow one does") {
      withCpu { ops =>
        val full = plain.run(ops)
        assert(plain.copy(window = Some(40)).run(ops).sameElements(full))
        assert(!plain.copy(window = Some(8)).run(ops).sameElements(full))
      }
    }
  }
}
