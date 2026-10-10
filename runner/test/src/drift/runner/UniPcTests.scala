package drift.runner

import utest.*

import drift.runner.diffusion.{HiDreamO1Pipeline, UniPc, UniPcBlend}

import java.nio.file.Files

/** UniPC against the official `FlowUniPCMultistepScheduler`, driven as HiDream
  * O1's `models/pipeline.py` drives it over a small synthetic denoiser
  * (`fixtures/unipc.py`): its noise levels, the whole timesteps the loop reads
  * off them, and every step's sample.
  */
object UniPcTests extends TestSuite {

  final private case class Case(
      steps: Int,
      shift: Double,
      sigmas: Seq[Double],
      timesteps: Seq[Int],
      start: Seq[Double],
      samples: Seq[Seq[Double]]
  )

  private lazy val cases: Seq[Case] =
    ujson
      .read(Files.readString(Fixtures.path("unipc.json")))("cases")
      .arr
      .toSeq
      .map(c =>
        Case(
          c("steps").num.toInt,
          c("shift").num,
          c("sigmas").arr.toSeq.map(_.num),
          c("timesteps").arr.toSeq.map(_.num.toInt),
          c("start").arr.toSeq.map(_.num),
          c("samples").arr.toSeq.map(_.arr.toSeq.map(_.num))
        )
      )

  /** The fixture's denoiser: a clean image of the sample and the timestep. */
  private def denoised(z: Array[Double], t: Double): Array[Double] =
    z.map(v => 0.9 * math.tanh(0.7 * v + 0.3 * t))

  private def blended(
      blend: UniPcBlend,
      sample: Array[Double],
      latest: Array[Double],
      earlier: Array[Double],
      current: Array[Double]
  ): Array[Double] =
    sample.indices.map { i =>
      def term(weight: Double, values: Array[Double]) =
        if (weight == 0) 0.0 else weight * values(i)
      term(blend.sample, sample) + term(blend.latest, latest) +
        term(blend.earlier, earlier) + term(blend.current, current)
    }.toArray

  /** Every step's sample, the solver driven as `HiDreamO1Pipeline` drives it.
    */
  private def samples(
      sigmas: IndexedSeq[Float],
      start: Seq[Double]
  ): Seq[Array[Double]] = {
    val solver = new UniPc(sigmas)
    var z = start.toArray
    var (latest, earlier, last) =
      (Array.empty[Double], Array.empty[Double], Array.empty[Double])
    (0 until sigmas.size - 1).map { i =>
      val level = HiDreamO1Pipeline.wholeLevel(sigmas(i))
      val reach = sigmas(i) / math.max(level, HiDreamO1Pipeline.MinimumSigma)
      val estimate = denoised(z, 1.0 - level)
      val clean =
        z.indices.map(k => z(k) + reach * (estimate(k) - z(k))).toArray
      solver
        .corrector(i)
        .foreach(c => z = blended(c, last, latest, earlier, clean))
      earlier = latest
      latest = clean
      last = z
      z = blended(solver.predictor(i), z, latest, earlier, clean)
      z
    }
  }

  val tests = Tests {
    test("the noise levels are the scheduler's, shifted twice") {
      cases.foreach { c =>
        val sigmas = UniPc.sigmas(c.steps, c.shift)
        assert(sigmas.size == c.sigmas.size)
        assert(sigmas.zip(c.sigmas).forall((a, e) => math.abs(a - e) < 1e-6))
      }
    }
    test("the model's level is the scheduler's whole timestep") {
      cases.foreach { c =>
        val whole = UniPc
          .sigmas(c.steps, c.shift)
          .init
          .map(s => math.round(HiDreamO1Pipeline.wholeLevel(s) * 1000))
        assert(whole == c.timesteps)
      }
    }
    test("50, 8, 3 and 1 steps land where the official scheduler does") {
      cases.foreach { c =>
        val made = samples(UniPc.sigmas(c.steps, c.shift), c.start)
        val worst = made
          .zip(c.samples)
          .flatMap((actual, expected) =>
            actual
              .zip(expected)
              .map((a, e) => math.abs(a - e) / (1 + math.abs(e)))
          )
          .max
        println(f"  ${c.steps} steps at shift ${c.shift}: worst ${worst}%.2e")
        assert(made.size == c.samples.size, worst < 2e-4)
      }
    }
    test("the order is 2 between a first and a last step of 1") {
      val solver = new UniPc(UniPc.sigmas(6, 3.0))
      assert((0 until 6).map(solver.order) == Seq(1, 2, 2, 2, 2, 1))
      // img2img from step 3: no history before it
      val late = new UniPc(UniPc.sigmas(6, 3.0), first = 3)
      assert(
        (3 until 6).map(late.order) == Seq(1, 2, 1),
        late.corrector(3).isEmpty,
        late.corrector(4).isDefined
      )
    }
    test("a list of levels that starts at 1 still steps") {
      val solver = new UniPc(IndexedSeq(1f, 0.75f, 0.5f, 0.25f, 0f))
      val weights = (0 until 4).flatMap { i =>
        val p = solver.predictor(i)
        solver
          .corrector(i)
          .toSeq
          .flatMap(c => Seq(c.sample, c.latest, c.earlier, c.current)) ++ Seq(
          p.sample,
          p.latest,
          p.earlier
        )
      }
      assert(weights.forall(w => !w.isNaN && !w.isInfinite))
    }
  }
}
