package drift.runner.server

import drift.runner.decode.ChatEngine
import drift.runner.native.HipRuntime
import drift.runner.ops.{HipOps, MatVecInputs}

/** The runner as drift launches it: `llama-server`'s flags and API (`specs/42`,
  * step 7). drift installs a folder whose `llama-server` starts this main (the
  * backend's `RunnerFiles`, `specs/43`).
  */
object Main {

  val Version = "drift runner 0.1.0"

  /** The GGUF architectures it chats with (`models/Models`), as drift's
    * architectures name their model kind (`specs/43`).
    */
  val ModelKinds: Seq[String] =
    Seq("qwen3", "qwen3vl", "qwen35", "qwen35moe", "qwen4exp")

  def main(arguments: Array[String]): Unit = {
    if (arguments.contains("--version")) RunnerIdentity.version(Version)
    if (arguments.contains("--model-kinds"))
      RunnerIdentity.modelKinds(ModelKinds)
    if (arguments.contains("--help") || arguments.contains("-h")) {
      println(
        s"$Version: llama-server's API on drift's own engine\n  -m FILE  -c CONTEXT  --host HOST  --port PORT"
      )
      sys.exit(0)
    }
    ServerOptions.parse(arguments.toSeq) match {
      case Left(problem) =>
        System.err.println(s"error: $problem")
        sys.exit(1)
      case Right(options) =>
        println(
          s"$Version: ${options.model} with a context of ${options.context}"
        )
        options.notes.foreach(println)
        val hip = HipRuntime.fromEnvironment()
        println(s"device: ${hip.deviceName} (${hip.rocmRoot})")
        val ops = new HipOps(hip, MatVecInputs.Float)
        val server = new ChatServer(
          options,
          () => {
            val started = System.nanoTime()
            val engine = new ChatEngine(
              ops,
              options.model,
              options.draftModel,
              options.visionModel,
              options.context,
              options.drafts
            )
            println(
              f"model loaded in ${(System.nanoTime() - started) / 1e9}%.1f s; listening on http://${options.host}:${options.port}"
            )
            engine
          }
        )
        server.start()
        println(
          s"server is listening on http://${options.host}:${options.port}"
        )
    }
  }
}
