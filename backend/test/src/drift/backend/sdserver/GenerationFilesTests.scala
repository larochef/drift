package drift.backend.sdserver

import utest.*

import java.nio.file.Files
import java.util.Base64

/** A video model's inputs are clips and sounds as well as images (`specs/42`,
  * step 14). Worth testing because the round trip is what reuse depends on: an
  * input written from a data URL must be served back with the MIME type it came
  * with, or the reloaded recipe sends the runner an `application/octet-stream`
  * it cannot place.
  */
object GenerationFilesTests extends TestSuite {

  private val bytes = Array[Byte](1, 2, 3, 4)

  private def dataUrl(mime: String): String =
    s"data:$mime;base64,${Base64.getEncoder.encodeToString(bytes)}"

  val tests = Tests {
    test("inputs keep their medium from data URL to served file") {
      val root = Files.createTempDirectory("drift-inputs")
      val files = GenerationFiles(root)
      List(
        "video/webm" -> "webm",
        "video/mp4" -> "mp4",
        "video/quicktime" -> "mov",
        "audio/wav" -> "wav",
        "audio/mpeg" -> "mp3",
        "audio/flac" -> "flac",
        "image/jpeg" -> "jpeg",
        "image/png" -> "png"
      ).foreach { (mime, extension) =>
        val url = files.externalizeInput(
          "g1-1",
          0L,
          "reference0",
          dataUrl(mime),
          scratch = true
        )
        assert(url == s"/api/outputs/scratch/g1-1-reference0.$extension")
        assert(GenerationManager.mimeTypeFor(extension) == mime)
        assert(
          Files
            .readAllBytes(root.resolve(s"scratch/g1-1-reference0.$extension"))
            .sameElements(bytes)
        )
      }
    }

    test("a browser's alias names the same file type") {
      assert(
        GenerationFiles.decodeMediaData(dataUrl("audio/x-wav"))._2 == "wav"
      )
      assert(GenerationFiles.decodeMediaData(dataUrl("audio/mp3"))._2 == "mp3")
    }

    test("a bare payload or an unknown type is taken for a PNG") {
      assert(
        GenerationFiles
          .decodeMediaData(Base64.getEncoder.encodeToString(bytes))
          ._2 == "png"
      )
      assert(
        GenerationFiles
          .decodeMediaData(dataUrl("application/octet-stream"))
          ._2 == "png"
      )
    }
  }
}
