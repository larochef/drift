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
