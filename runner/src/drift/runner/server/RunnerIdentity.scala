package drift.runner.server

import drift.runner.native.{HipRuntime, KernelModule}

import scala.util.control.NonFatal

/** What the runner answers drift's validation with (`specs/43`): its version,
  * proven on this computer, and the model kinds it runs.
  */
object RunnerIdentity {

  /** Prints `version` and the GPU, and exits: 0 when the GPU loads the runner's
    * gfx1151 kernels, 1 with the reason otherwise (another GPU, no TheRock
    * tree).
    */
  def version(version: String): Nothing =
    try {
      val hip = HipRuntime.fromEnvironment()
      val name = hip.deviceName
      try new KernelModule(hip, "elementwise").close()
      catch {
        case NonFatal(error) =>
          println(
            s"$version: this GPU ($name) cannot run the runner's gfx1151 kernels: ${error.getMessage}"
          )
          sys.exit(1)
      }
      println(s"$version (Strix Halo, gfx1151) on $name")
      sys.exit(0)
    } catch {
      case NonFatal(error) =>
        println(s"$version: no usable GPU: ${error.getMessage}")
        sys.exit(1)
    }

  /** Prints `kinds`, one per line, and exits. */
  def modelKinds(kinds: Seq[String]): Nothing = {
    kinds.foreach(println)
    sys.exit(0)
  }
}
