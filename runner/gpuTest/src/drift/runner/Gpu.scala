package drift.runner

import drift.runner.native.HipRuntime

/** The one HIP runtime of the test process. */
object Gpu {
  lazy val hip: HipRuntime = HipRuntime.fromEnvironment()
}
