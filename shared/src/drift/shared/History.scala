package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** The gallery's read side over the generation sidecars
  * (`specs/12-gallery.md`). The `outputs/<date>/` layout is the index: history
  * is paged by date directory, newest first, and the frontend loads one day at
  * a time.
  */

/** One date directory under the outputs root and how many sidecars it holds.
  */
case class HistoryDay(date: String, count: Int)
object HistoryDay {
  given Schema[HistoryDay] = Schema.derived
  given JsonValueCodec[List[HistoryDay]] = JsonCodecMaker.make
  given JsonValueCodec[Option[HistoryDay]] = JsonCodecMaker.make
}

// `base` in Api.scala is private to that file (tracked as bug 16); the same
// prefix is rebuilt here so these endpoints live under /api like the rest.
private val historyBase = endpoint.in("api")

/** Every day that has recorded generations, newest first. */
val listHistoryDays: PublicEndpoint[Unit, Unit, List[HistoryDay], Any] =
  historyBase.get.in("history").out(jsonBody[List[HistoryDay]])

/** One day's recorded generations, newest first. A sidecar that no longer
  * decodes is skipped (and logged) rather than failing the day.
  */
val listHistoryDay: PublicEndpoint[String, Unit, List[Generation], Any] =
  historyBase.get
    .in("history" / path[String]("date"))
    .out(jsonBody[List[Generation]])

/** Which day holds one generation, or nothing when no generation by that id is
  * recorded. A detail view opened straight from its URL — a refresh, a link,
  * the back button — sits on a page that has loaded only the newest days, and
  * this is how it finds the day to load for the one it was asked to show
  * (François, 2026-09-20).
  */
val findHistoryGenerationDay
    : PublicEndpoint[String, Unit, Option[HistoryDay], Any] =
  historyBase.get
    .in("history" / "generation" / path[String]("generationId"))
    .out(jsonBody[Option[HistoryDay]])

/** Deletes one generation: its output files, its persisted input images and its
  * sidecar. Answers false when nothing by that id exists under that day.
  */
val deleteHistoryGeneration
    : PublicEndpoint[(String, String), Unit, Boolean, Any] =
  historyBase.delete
    .in("history" / path[String]("date") / path[String]("generationId"))
    .out(jsonBody[Boolean])

/** An image from outside drift, brought into the gallery so it can be upscaled,
  * redrawn or edited like one drift made
  * (`specs/30-gallery-ergonomics-and-image-import.md`). `data` is a
  * `data:image/...;base64,` URL, the shape input images already travel in.
  */
case class ImageImport(fileName: String, data: String)
object ImageImport {
  given Schema[ImageImport] = Schema.derived
  given JsonValueCodec[ImageImport] = JsonCodecMaker.make
}

/** Imports one image as a gallery entry of its own, kind "import", filed under
  * today. The failure is why it was refused — not a PNG or JPEG, unreadable.
  */
val importHistoryImage: PublicEndpoint[ImageImport, String, Generation, Any] =
  historyBase.post
    .in("history" / "imports")
    .in(LargeJsonBody[ImageImport])
    .errorOut(stringBody)
    .out(jsonBody[Generation])

/** Imports one video the same way. The file is the body itself, written to disk
  * as it arrives: a video inside a JSON string would be held whole in memory
  * several times over, on both sides. The failure is why it was refused — not a
  * WebM, MP4, MOV or Matroska file.
  */
val importHistoryVideo
    : PublicEndpoint[(String, TapirFile), String, Generation, Any] =
  historyBase.post
    .in("history" / "imports" / "videos")
    .in(query[String]("fileName"))
    .in(fileBody)
    .errorOut(stringBody)
    .out(jsonBody[Generation])

/** Gallery entries given to a project, or taken out of any (`projectId` none).
  * `generations` names each by the day it is filed under and its id.
  */
case class GenerationMove(
    generations: List[GenerationReference],
    projectId: Option[String]
)
object GenerationMove {
  given Schema[GenerationMove] = Schema.derived
  given JsonValueCodec[GenerationMove] = JsonCodecMaker.make
}

case class GenerationReference(date: String, generationId: String)
object GenerationReference {
  given Schema[GenerationReference] = Schema.derived

  def of(generation: Generation): Option[GenerationReference] =
    generation.outputs.headOption
      .map(output => GenerationReference(output.date, generation.id))
}

/** Moves generations to a project (`specs/19-projects-and-prompt-versions.md`):
  * orphans — imports, the Sandbox's — or another project's. What was derived
  * from one follows it. Answers every entry it rewrote; the failure is why
  * nothing moved — the project does not exist.
  */
val moveHistoryGenerations
    : PublicEndpoint[GenerationMove, String, List[Generation], Any] =
  historyBase.post
    .in("history" / "moves")
    .in(jsonBody[GenerationMove])
    .errorOut(stringBody)
    .out(jsonBody[List[Generation]])
