package drift.runner

import utest.*

import drift.runner.decode.{
  Drafter,
  Generated,
  Generator,
  Prompt,
  Sampler,
  Sampling,
  Speculation
}
import drift.runner.models.{CausalModel, Models, Qwen3}
import drift.runner.ops.{CpuOps, Ops}
import drift.runner.state.Snapshot
import drift.runner.text.{ByteLevel, TokenizerJson}

/** The generator on the tiny models: a prefill cut into chunks, then decoding
  * step by step, must give the tokens a naive loop gives by re-running the
  * whole sequence each time; drafted and verified, and resumed from what the
  * sequence holds, the same tokens again.
  */
object GeneratorTests extends TestSuite {

  private val Seeded = Sampling(0.8f, 0, 1f, 0f, 7L)

  private def generate(
      ops: Ops,
      model: CausalModel,
      speculation: Option[Speculation],
      sampling: Sampling,
      prompts: Seq[Array[Int]],
      turnStarts: Set[Int] = Set.empty
  ): Seq[Generated] = {
    val generator = new Generator(
      ops,
      model,
      tokenizer,
      context = 64,
      pageSize = 16,
      prefillChunk = 4,
      speculation,
      turnStarts = turnStarts
    )
    try
      prompts.map(prompt =>
        generator
          .generate(prompt, 12, new Sampler(sampling), Set.empty, _ => true)
      )
    finally generator.close()
  }

  /** Knows the answer (the plain run's tokens after `prompt`) and drafts it,
    * wrong on purpose at some positions: whole, partial and no acceptance.
    */
  final private class Oracle(prompt: Array[Int], answer: Seq[Int])
      extends Drafter {
    def observed(ids: Array[Int], start: Int, kept: Int): Unit = ()
    def propose(last: Int, position: Int, count: Int): Array[Int] =
      Array.tabulate(count) { j =>
        val index = position + 1 + j - prompt.length
        val right = answer.lift(index).getOrElse(0)
        if ((position + j) % 5 == 3) (right + 1) % 320 else right
      }
    def reset(): Unit = ()
    def snapshot(): Snapshot = new Snapshot {
      def restore(): Unit = ()
      def close(): Unit = ()
    }
    def close(): Unit = ()
  }

  private def open(ops: Ops, folder: String) =
    Models.open(
      ops,
      Fixtures.path(s"tiny/$folder/model.safetensors"),
      TinyModelCase.draftModel(folder)
    )

  /** A byte-level tokenizer of the tiny model's 320 ids, so that ids decode. */
  private lazy val tokenizer = {
    val tokens = (0 until 320).map(i =>
      if (i < 256) ByteLevel.characterOf(i).toString
      else s"${ByteLevel.characterOf(i - 256)}${ByteLevel.characterOf(i - 190)}"
    )
    TokenizerJson.read(
      ujson.Obj(
        "model" -> ujson.Obj(
          "type" -> "BPE",
          "vocab" -> ujson.Obj.from(
            tokens.zipWithIndex.map((t, i) => t -> ujson.Num(i))
          ),
          "merges" -> ujson.Arr()
        ),
        "normalizer" -> ujson.Null,
        "pre_tokenizer" -> ujson.Obj(
          "type" -> "Sequence",
          "pretokenizers" -> ujson.Arr(
            ujson.Obj(
              "type" -> "Split",
              "pattern" -> ujson.Obj("Regex" -> "\\s+|\\S+"),
              "behavior" -> "Isolated",
              "invert" -> false
            ),
            ujson.Obj(
              "type" -> "ByteLevel",
              "add_prefix_space" -> false,
              "use_regex" -> false
            )
          )
        ),
        "post_processor" -> ujson.Null,
        "added_tokens" -> ujson.Arr()
      ),
      "tiny"
    )
  }

  /** On `ops` (closed after): drafts right and wrong give the plain run's
    * tokens.
    */
  def draftsChangeNothing(ops: Ops): Unit = {
    for {
      folder <- Seq("qwen3", "qwen35", "qwen35moe", "qwen4exp")
      sampling <- Seq(Sampling.Greedy, Seeded)
    } {
      val model = open(ops, folder)
      try {
        val prompt = TinyModelCase.Qwen3.ids.take(9)
        val Seq(plain) = generate(ops, model, None, sampling, Seq(prompt))
        val oracle =
          Speculation(3, _ => new Oracle(prompt, plain.ids))
        val Seq(drafted) =
          generate(ops, model, Some(oracle), sampling, Seq(prompt))
        assert(drafted.ids == plain.ids)
        assert(drafted.accepted > 0, drafted.accepted < drafted.drafted)
      } finally model.close()
    }
    ops.close()
  }

  /** On `ops` (closed after): the MTP layer's drafts give the plain run's
    * tokens.
    */
  def multiTokenChangesNothing(ops: Ops): Unit =
    try
      Seq(
        "qwen35" -> TinyModelCase.Qwen35,
        "qwen35moe" -> TinyModelCase.Qwen35Moe,
        "qwen4exp" -> TinyModelCase.Qwen4Exp
      ).foreach((folder, fixture) =>
        multiTokenChangesNothing(ops, folder, fixture)
      )
    finally ops.close()

  private def multiTokenChangesNothing(
      ops: Ops,
      folder: String,
      fixture: TinyModelCase
  ): Unit = {
    val model = open(ops, folder)
    try {
      val prompt = fixture.ids.take(11)
      val speculation = Speculation.multiToken(ops, model, 2, prefillChunk = 4)
      assert(speculation.isDefined)
      Seq(Sampling.Greedy, Seeded).foreach { sampling =>
        val Seq(plain) = generate(ops, model, None, sampling, Seq(prompt))
        val Seq(drafted) =
          generate(ops, model, speculation, sampling, Seq(prompt))
        assert(drafted.ids == plain.ids, drafted.drafted > 0)
        println(
          s"  accepted ${drafted.accepted} of ${drafted.drafted} random-weight drafts"
        )
      }
    } finally model.close()
  }

  val tests = Tests {
    test("chunked prefill and decoding give the naive loop's tokens") {
      val ops = new CpuOps
      val model = Qwen3.open(ops, Fixtures.path("tiny/qwen3/model.safetensors"))
      try {
        val prompt = TinyModelCase.Qwen3.ids.take(11)
        val generator = new Generator(
          ops,
          model,
          tokenizer,
          context = 64,
          pageSize = 16,
          prefillChunk = 4,
          speculation = None
        )
        val generated = generator.generate(
          prompt,
          8,
          new Sampler(Sampling.Greedy),
          Set.empty,
          _ => true
        )
        generator.close()

        // naive: the whole sequence again at every step, a fresh cache each time
        var sequence = prompt.toVector
        (1 to 8).foreach { _ =>
          val fresh = model.newSequence(64, 16)
          fresh.reserve(sequence.length)
          val logits = ops.toFloats(
            model.forward(sequence.toArray, 0, fresh, allLogits = false)
          )
          sequence :+= logits.indices.maxBy(logits(_))
        }
        assert(generated.ids == sequence.drop(prompt.length))
        assert(
          generated.text == tokenizer.decode(generated.ids, skipSpecial = true)
        )
      } finally {
        model.close()
        ops.close()
      }
    }
    test("drafts, right or wrong, change no token") {
      draftsChangeNothing(new CpuOps)
    }
    test("the MTP layer's drafts change no token") {
      multiTokenChangesNothing(new CpuOps)
    }
    test(
      "a prompt resumes from the last one and its reply, or a turn's start"
    ) {
      resumes("qwen35moe", TinyModelCase.Qwen35Moe)
      resumes("qwen4exp", TinyModelCase.Qwen4Exp)
    }
  }

  private def resumes(folder: String, fixture: TinyModelCase): Unit = {
    val ops = new CpuOps
    val model = open(ops, folder)
    try {
      val prompt = fixture.ids.take(7)
      val speculation =
        Speculation.multiToken(ops, model, 2, prefillChunk = 4)
      Seq(None, speculation).foreach { drafting =>
        val Seq(first) =
          generate(ops, model, drafting, Sampling.Greedy, Seq(prompt))
        val continued = prompt ++ first.ids ++ Array(5, 6)
        val branched = prompt ++ Array(9, 10, 11)
        // the reply kept, or only the prompt (the checkpoint at its end)
        val Seq(_, again) =
          generate(
            ops,
            model,
            drafting,
            Sampling.Greedy,
            Seq(prompt, continued)
          )
        val Seq(_, other) =
          generate(
            ops,
            model,
            drafting,
            Sampling.Greedy,
            Seq(prompt, branched)
          )
        // the reply's last token was never run: all but it are reused
        assert(again.reusedTokens == prompt.length + first.ids.length - 1)
        assert(other.reusedTokens == prompt.length)
        val Seq(freshAgain) =
          generate(ops, model, drafting, Sampling.Greedy, Seq(continued))
        val Seq(freshOther) =
          generate(ops, model, drafting, Sampling.Greedy, Seq(branched))
        assert(again.ids == freshAgain.ids, other.ids == freshOther.ids)
        // what follows a turn's start rewritten: back to before that token
        val opening = prompt(4)
        val rewritten = prompt.take(5) ++ Array(9, 10, 11)
        val Seq(_, turned) = generate(
          ops,
          model,
          drafting,
          Sampling.Greedy,
          Seq(prompt, rewritten),
          Set(opening)
        )
        val Seq(freshTurned) =
          generate(ops, model, drafting, Sampling.Greedy, Seq(rewritten))
        // checkpoints before each opening and at the prompt's end
        val shared = Prompt
          .text(rewritten)
          .sharedPrefix(prompt ++ first.ids.init, Nil)
        val back =
          (prompt.indices.filter(prompt(_) == opening) :+ prompt.length)
            .filter(_ <= shared)
            .max
        assert(back >= 4, turned.reusedTokens == back)
        assert(turned.ids == freshTurned.ids)
      }
    } finally {
      model.close()
      ops.close()
    }
  }
}
