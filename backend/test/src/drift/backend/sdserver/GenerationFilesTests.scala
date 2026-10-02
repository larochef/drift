package drift.backend.sdserver

import drift.shared.InputSource

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

    // `specs/50-inputs-from-the-gallery.md`: a picked entry arrives as the URL
    // it is served from; the bytes sd-server takes and the way back to the
    // entry are both read off the outputs.
    test("a gallery output's URL becomes its bytes and names its entry") {
      val root = Files.createTempDirectory("drift-served")
      val day = Files.createDirectories(root.resolve("2026-10-02"))
      Files.write(day.resolve("g7-1-0.png"), bytes)
      Files.write(day.resolve("g7-1-1.png"), bytes)
      Files.write(day.resolve("g7-1-init.png"), bytes)
      // A sidecar as it was written before inputs had sources: no such field.
      Files.writeString(
        day.resolve("g7-1.json"),
        """{"id":"g7-1","session_id":"s","run_configuration_id":"c",
          |"kind":"img_gen","status":"Completed","submitted_at":1,
          |"outputs":[
          |{"date":"2026-10-02","file_name":"g7-1-0.png",
          | "url":"/api/outputs/2026-10-02/g7-1-0.png",
          | "mime_type":"image/png","format":"png"},
          |{"date":"2026-10-02","file_name":"g7-1-1.png",
          | "url":"/api/outputs/2026-10-02/g7-1-1.png",
          | "mime_type":"image/png","format":"png","index":1}]}""".stripMargin
      )
      val files = GenerationFiles(root)

      val output = files.servedInput("/api/outputs/2026-10-02/g7-1-1.png")
      assert(output.exists(_.isRight))
      val served = output.get.toOption.get
      assert(served.data == dataUrl("image/png"))
      assert(
        served.source.contains(
          InputSource("", "g7-1", 1, "2026-10-02", "g7-1-1.png")
        )
      )

      // An input kept beside the outputs is a file, and no entry's output.
      val input = files.servedInput("/api/outputs/2026-10-02/g7-1-init.png")
      assert(input.exists(_.exists(_.source.isEmpty)))

      // Bytes are left alone; a file that is gone is said to be.
      assert(files.servedInput(dataUrl("image/png")).isEmpty)
      assert(
        files.servedInput("/api/outputs/2026-10-02/gone.png").exists(_.isLeft)
      )
      assert(files.servedInput("/api/outputs/../secret.png").isEmpty)
    }
  }
}
