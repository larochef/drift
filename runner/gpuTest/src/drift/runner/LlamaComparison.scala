package drift.runner

import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.file.{Path, Paths}
import java.time.LocalDateTime

import scala.util.Try

import drift.runner.models.Models
import drift.runner.ops.{HipOps, MatVecInputs}
import drift.runner.text.ChatTemplate

/** The runner against llama.cpp on the same GGUF (`specs/42`, Testing, layer
  * 5): llama-server generates greedily with its top-10 log-probabilities per
  * step; the runner is fed the same tokens and its own top-10 are compared at
  * every position.
  *
  * `./mill runner.gpuTest.runMain drift.runner.LlamaComparison <model.gguf> <llama-server> [rocm lib]`
  */
object LlamaComparison {

  def main(arguments: Array[String]): Unit = {
    val (model, server) = (Paths.get(arguments(0)), Paths.get(arguments(1)))
    val tokenizer = TokenizerFiles.ggufTokenizer(model)
    val (file, mapped) = drift.runner.formats.Gguf.open(model)
    val template = new ChatTemplate(
      file.string("tokenizer.chat_template"),
      () => LocalDateTime.now()
    )
    mapped.close()
    val prompt = tokenizer.encode(
      template.render(
        ujson.Arr(
          ujson.Obj(
            "role" -> "user",
            "content" -> "List three facts about the Moon, one line each."
          )
        ),
        None,
        addGenerationPrompt = true,
        Map("enable_thinking" -> ujson.False)
      ),
      addSpecial = true
    )
    val steps = llama(model, server, arguments.lift(2), prompt, 24)
    println(
      s"llama.cpp generated ${steps.size} tokens: ${tokenizer.decode(steps.map(_._1), skipSpecial = false)}"
    )

    val ops = new HipOps(Gpu.hip, MatVecInputs.Int8)
    val qwen = Models.open(ops, model, None)
    val pageSize = 64
    val pageCount = 8
    val sequence = qwen.newSequence(pageSize * pageCount, pageSize)
    sequence.reserve(prompt.length + steps.size)
    // chunk = 8: the prompt takes the matrix-vector path (int8 x), not the F16 GEMM
    val chunk = arguments.lift(3).map(_.toInt).getOrElse(prompt.length)
    var logits = Array.emptyFloatArray
    prompt.grouped(chunk).zipWithIndex.foreach { (part, i) =>
      logits = ops.toFloats(
        qwen.forward(part, i * chunk, sequence, allLogits = false)
      )
    }
    var sameArgmax = 0
    var worstGap = 0.0
    steps.zipWithIndex.foreach { case ((chosen, theirs), step) =>
      val ours = logSoftmax(logits)
      val ourTop = ours.indices.sortBy(-ours(_)).take(10)
      if (ourTop.head == theirs.head._1) sameArgmax += 1
      // log-probability differences on llama.cpp's top 10
      val gap = theirs.map((id, logprob) => math.abs(ours(id) - logprob)).max
      worstGap = math.max(worstGap, gap)
      println(
        f"  step $step%2d: llama ${show(tokenizer, theirs.head._1)}%-14s ours ${show(tokenizer, ourTop.head)}%-14s " +
          f"top-10 overlap ${ourTop.toSet.intersect(theirs.map(_._1).toSet).size}%2d, worst log-prob gap $gap%.3f, " +
          f"chosen ${theirs.head._2}%.3f vs ${ours(theirs.head._1)}%.3f, sum of llama's top-10 p ${theirs.map(t => math.exp(t._2)).sum}%.3f vs ours ${theirs.map(t => math.exp(ours(t._1))).sum}%.3f"
      )
      logits = ops.toFloats(
        qwen.forward(
          Array(chosen),
          prompt.length + step,
          sequence,
          allLogits = false
        )
      )
    }
    println(
      f"same most likely token at $sameArgmax/${steps.size} steps, worst log-prob gap $worstGap%.3f"
    )
    qwen.close()
    ops.close()
  }

  private def show(tokenizer: drift.runner.text.Tokenizer, id: Int) =
    ujson.write(tokenizer.decode(Seq(id), skipSpecial = false))

  private def logSoftmax(logits: Array[Float]): Array[Double] = {
    val largest = logits.max.toDouble
    val total = math.log(logits.map(l => math.exp(l - largest)).sum) + largest
    logits.map(_ - total)
  }

  /** Each step's chosen token and llama.cpp's top-10 (id, log-probability). */
  private def llama(
      model: Path,
      server: Path,
      rocm: Option[String],
      prompt: Array[Int],
      count: Int
  ): Seq[(Int, Seq[(Int, Double)])] = {
    val port = 18093
    val builder = new ProcessBuilder(
      server.toString,
      "-m",
      model.toString,
      "-ngl",
      "99",
      "-c",
      "2048",
      "--port",
      port.toString,
      "-fa",
      "on"
    )
    val libraries = (server.getParent.toString +: rocm.toSeq).mkString(":")
    builder.environment().put("LD_LIBRARY_PATH", libraries)
    builder
      .redirectErrorStream(true)
      .redirectOutput(ProcessBuilder.Redirect.DISCARD)
    val process = builder.start()
    try {
      val client = HttpClient.newHttpClient()
      def get(path: String) = Try(
        client.send(
          HttpRequest
            .newBuilder(URI.create(s"http://127.0.0.1:$port$path"))
            .build(),
          HttpResponse.BodyHandlers.ofString()
        )
      )
      var waited = 0
      while (
        !get("/health").toOption.exists(_.statusCode() == 200) && waited < 240
      ) { Thread.sleep(500); waited += 1 }
      val body = ujson.Obj(
        "prompt" -> ujson.Arr(prompt.map(ujson.Num(_))*),
        "n_predict" -> count,
        "n_probs" -> 10,
        "temperature" -> 0,
        "cache_prompt" -> false
      )
      val response = client.send(
        HttpRequest
          .newBuilder(URI.create(s"http://127.0.0.1:$port/completion"))
          .POST(HttpRequest.BodyPublishers.ofString(ujson.write(body)))
          .build(),
        HttpResponse.BodyHandlers.ofString()
      )
      ujson.read(response.body())("completion_probabilities").arr.toSeq.map {
        step =>
          step("id").num.toInt -> step("top_logprobs").arr.toSeq
            .map(p => p("id").num.toInt -> p("logprob").num)
      }
    } finally process.destroy()
  }
}
