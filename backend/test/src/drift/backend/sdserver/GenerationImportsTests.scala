package drift.backend.sdserver

import drift.shared.*
import utest.*

import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.nio.file.Files
import java.util.Base64
import javax.imageio.ImageIO

import com.github.plokhotnyuk.jsoniter_scala.core.readFromArray

/** Importing an image into the gallery (`specs/30`). Worth testing because the
  * entry is only useful if the rest of drift takes it for a generation: the id
  * the history matches on, the file naming its deletes rely on, a sidecar that
  * reads back.
  */
object GenerationImportsTests extends TestSuite {

  private def dataUrl(format: String, mime: String): String = {
    val out = ByteArrayOutputStream()
    ImageIO.write(BufferedImage(8, 6, BufferedImage.TYPE_INT_RGB), format, out)
    s"data:$mime;base64,${Base64.getEncoder.encodeToString(out.toByteArray)}"
  }

  val tests = Tests {
    test("an import is a gallery entry the history reads back") {
      val root = Files.createTempDirectory("drift-imports")
      val imported = GenerationImports(root)
        .importImage(ImageImport("holiday.jpg", dataUrl("jpeg", "image/jpeg")))
        .toOption
        .get
      val output = imported.outputs.head
      assert(imported.kind == "import")
      assert(GenerationHistory.isGenerationId(imported.id))
      assert(output.fileName == s"${imported.id}-0.jpeg")
      assert(output.mimeType == "image/jpeg")
      assert(Files.isRegularFile(root.resolve(output.date).resolve(output.fileName)))
      val sidecar = readFromArray[Generation](
        Files.readAllBytes(root.resolve(output.date).resolve(s"${imported.id}.json"))
      )
      assert(sidecar == imported)
      assert(sidecar.importedFileName.contains("holiday.jpg"))
    }
    test("the format comes from the bytes, not the name or the mime type") {
      val root = Files.createTempDirectory("drift-imports")
      val imported = GenerationImports(root)
        .importImage(ImageImport("mislabelled.jpg", dataUrl("png", "image/jpeg")))
        .toOption
        .get
      assert(imported.outputs.head.format == "png")
    }
    test("anything but PNG or JPEG is refused and leaves nothing behind") {
      val root = Files.createTempDirectory("drift-imports")
      val refused = GenerationImports(root)
        .importImage(ImageImport("animation.gif", dataUrl("gif", "image/gif")))
      assert(refused.isLeft)
      assert(Files.list(root).count() == 0)
    }
  }
}
