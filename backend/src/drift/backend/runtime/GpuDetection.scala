package drift.backend.runtime

import java.nio.charset.StandardCharsets
import java.nio.file.*
import scala.jdk.CollectionConverters.*
import scala.util.Try

/** What GPU the machine has, as far as picking a first runtime needs to know
  * (`specs/46-starter-configurations.md`).
  */
object GpuDetection {

  /** Where the amdgpu kernel driver lists its compute nodes: one directory per
    * node, the CPU's reporting target version 0.
    */
  val KfdNodes: Path = Paths.get("/sys/class/kfd/kfd/topology/nodes")

  /** The gfx target of the first AMD GPU the ROCm kernel driver exposes, read
    * off its `gfx_target_version` (11.5.1 is written 110501, and is gfx1151).
    */
  def amdGfxTarget(nodes: Path = KfdNodes): Option[String] =
    Try {
      val listing = Files.list(nodes)
      try
        listing
          .iterator()
          .asScala
          .toList
          .sortBy(_.getFileName.toString)
          .flatMap(node => targetVersion(node.resolve("properties")))
          .find(_ > 0)
          .map(gfxName)
      finally listing.close()
    }.toOption.flatten

  private def targetVersion(properties: Path): Option[Int] =
    Try(Files.readAllLines(properties, StandardCharsets.UTF_8).asScala)
      .getOrElse(Nil)
      .collectFirst {
        case line if line.startsWith("gfx_target_version ") =>
          line.stripPrefix("gfx_target_version ").trim.toIntOption
      }
      .flatten

  /** 110501 → gfx1151, 90010 → gfx90a: the major in decimal, minor and stepping
    * as one hex digit each.
    */
  def gfxName(version: Int): String =
    s"gfx${version / 10000}${Integer.toHexString(version / 100 % 100)}${Integer
        .toHexString(version % 100)}"
}
