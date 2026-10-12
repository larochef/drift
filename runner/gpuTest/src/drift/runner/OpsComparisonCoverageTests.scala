package drift.runner

import drift.runner.ops.Ops

import java.io.DataInputStream
import java.lang.reflect.Modifier
import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*

import utest.*

/** Which operations of `Ops` no test here runs on both backends: the list is
  * written down, so an operation added without a comparison of the GPU with the
  * CPU fails this until it has one — or until it is added to the list on
  * purpose.
  *
  * An operation counts as compared when a test class that uses both `CpuOps`
  * and `HipOps` calls it: read from the compiled classes of these tests, whose
  * constant pools name every method they call.
  */
object OpsComparisonCoverageTests extends TestSuite {

  private val owners = Set(
    "drift/runner/ops/Ops",
    "drift/runner/ops/HipOps",
    "drift/runner/ops/CpuOps"
  )

  /** The methods a class file calls, each with the class it calls it on. */
  private def calls(file: Path): Set[(String, String)] = {
    val in = new DataInputStream(Files.newInputStream(file))
    try {
      in.readInt(); in.readUnsignedShort(); in.readUnsignedShort()
      val count = in.readUnsignedShort()
      val text = new Array[String](count)
      val classes = new Array[Int](count)
      val nameAndType = new Array[Int](count)
      val methods = scala.collection.mutable.ArrayBuffer.empty[(Int, Int)]
      var index = 1
      while (index < count) {
        in.readUnsignedByte() match {
          case 1                => text(index) = in.readUTF()
          case 3 | 4            => in.readInt()
          case 5 | 6            => in.readLong(); index += 1
          case 7                => classes(index) = in.readUnsignedShort()
          case 8 | 16 | 19 | 20 => in.readUnsignedShort()
          case 9       => in.readUnsignedShort(); in.readUnsignedShort()
          case 10 | 11 =>
            methods += ((in.readUnsignedShort(), in.readUnsignedShort()))
          case 12 =>
            nameAndType(index) = in.readUnsignedShort(); in.readUnsignedShort()
          case 15      => in.readUnsignedByte(); in.readUnsignedShort()
          case 17 | 18 => in.readUnsignedShort(); in.readUnsignedShort()
          case other   =>
            throw new IllegalStateException(s"$file: constant $other")
        }
        index += 1
      }
      methods
        .map((owner, named) => (text(classes(owner)), text(nameAndType(named))))
        .toSet
    } finally in.close()
  }

  /** Every operation a backend has to give: the abstract methods of `Ops`, but
    * what is no operation — its name, its work hook, what the compiler adds.
    */
  private val operations: Set[String] =
    classOf[Ops].getMethods.toSeq
      .filter(method => Modifier.isAbstract(method.getModifiers))
      .map(_.getName)
      .filterNot(name =>
        name.contains("$") || name == "name" || name == "onWork"
      )
      .toSet

  private def compared: Set[String] = {
    val root =
      Paths.get(getClass.getProtectionDomain.getCodeSource.getLocation.toURI)
    val walk = Files.walk(root)
    try
      walk.iterator.asScala.toSeq
        .filter(_.toString.endsWith(".class"))
        .map(calls)
        // a class that builds both backends is a comparison of them
        .filter(called =>
          called.exists((owner, name) =>
            owner == "drift/runner/ops/CpuOps" && name == "<init>"
          ) &&
            called.exists((owner, name) =>
              owner == "drift/runner/ops/HipOps" && name == "<init>"
            )
        )
        .flatMap(_.collect { case (owner, name) if owners(owner) => name })
        .toSet
    finally walk.close()
  }

  /** The operations no comparison of the two backends runs, as they stand: each
    * is reached on the GPU through a model's tiny-fixture test only.
    */
  private val Uncompared: Set[String] = Set(
    "averageDown",
    "columnMean",
    "depthwiseConv3x3",
    "duplicateUp",
    "gatedAddChunk",
    "joinHalves",
    "mapFile",
    "maxAbs",
    "modulateChunks",
    "patchesToPixels",
    "pixelsToPatches",
    "ropeTable",
    "rowGatedAdd",
    "scaledActivation",
    "scatterAddRows",
    "snake",
    "transpose",
    "unpackPatches",
    "upsample2x",
    "wideProducts"
  )

  val tests = Tests {
    test(
      "the operations no test compares on both backends are the listed ones"
    ) {
      val uncompared = operations -- compared
      val added = (uncompared -- Uncompared).toSeq.sorted
      val gone = (Uncompared -- uncompared).toSeq.sorted
      println(
        s"  ${operations.size} operations, ${uncompared.size} without a comparison: ${uncompared.toSeq.sorted.mkString(", ")}"
      )
      // an operation without a comparison that the list does not name
      assert(added.isEmpty)
      // an operation the list names that has one now, or is gone: the list is out of date
      assert(gone.isEmpty)
    }
  }
}
