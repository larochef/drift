package drift.backend.runtime

import drift.shared.Runtime

/** What an sd-cpp build can do that its sd-server does not report, read off the
  * release tag, `master-<build>-<sha>`.
  */
object SdCppBuilds {

  /** The first build whose sd-server leaves a reference image its own size when
    * the request says `auto_resize_ref_image: false`
    * (leejet/stable-diffusion.cpp#2004, fixed by #2011). Before it every
    * reference is scaled to the request's width × height while the request is
    * decoded, whatever its own shape — and the default still does that after.
    */
  val FirstKeepingReferenceSize: Int = 892

  private val ReleaseTag = """master-(\d+)-.*""".r

  /** Whether `runtime`'s sd-server keeps a reference its own size when asked;
    * drift's own runner does, and another tag that is not a numbered master
    * build says no.
    */
  def keepsReferenceSize(runtime: Runtime): Boolean =
    runtime.releaseTag match {
      case Runtime.DriftRunnerTag => true
      case ReleaseTag(build)      => build.toInt >= FirstKeepingReferenceSize
      case _                      => false
    }
}
