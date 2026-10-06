package drift.shared

import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.{
  readFromString,
  JsonValueCodec
}
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** What a redraw's tile is painted with when the assistant read the picture
  * (`specs/52-auto-redraw.md`): the tile, by its place in the picture, the
  * materials it shows as the prompt names them, and its own strength — zero
  * leaves the tile as it is.
  */
case class TileSettings(
    region: ImageRegion,
    prompt: String,
    strength: Double
)
object TileSettings {
  given Schema[TileSettings] = Schema.derived
}

/** One tile of a plan: its settings, the name it had on the picture the
  * assistant was shown, and what the assistant saw in it.
  */
case class PlannedTile(
    name: String,
    holds: List[String],
    settings: TileSettings
)
object PlannedTile {
  given Schema[PlannedTile] = Schema.derived
}

/** A part of the picture the assistant found broken, and what it should look
  * like instead. A proposal: it changes what the picture shows, so nothing
  * repaints it unless the user asks (François, 2026-10-05: eyes that glow may
  * be what the picture is about).
  */
case class PlannedRepair(
    defect: String,
    fix: String,
    region: ImageRegion
)
object PlannedRepair {
  given Schema[PlannedRepair] = Schema.derived
}

/** The assistant's reading of a picture for its redraw (`specs/52`): every tile
  * of the layout it was asked about, the repairs it proposes, and what had to
  * be put right in its answer, for the form to say.
  */
case class RedrawPlan(
    kind: String,
    tiles: List[PlannedTile],
    repairs: List[PlannedRepair],
    notes: List[String],
    /** How long the assistant took, in seconds. */
    seconds: Double
)

/** Asks for a plan of the redraw these fields describe: the tiles are the ones
  * a `RedrawRequest` with the same configuration, tile size and grid offset
  * cuts.
  */
case class RedrawPlanRequest(
    runConfigurationId: String,
    tileSize: Int = 1280,
    gridOffsetX: Int = 0,
    gridOffsetY: Int = 0,
    /** The assistant session to ask; none takes the first live one that reads
      * images.
      */
    sessionId: Option[String] = None,
    /** The diagnosis template (`specs/32-prompt-library.md`); none means the
      * built-in.
      */
    templateId: Option[String] = None
)

object RedrawPlanRequest {
  given JsonValueCodec[RedrawPlanRequest] = JsonCodecMaker.make
  given Schema[RedrawPlanRequest] = Schema.derived
}

object RedrawPlan {
  given JsonValueCodec[RedrawPlan] = JsonCodecMaker.make
  given Schema[RedrawPlan] = Schema.derived

  /** The longest side, in px, of the picture the assistant is shown: at 1024 a
    * 6 × 4 grid came back with 28 cells, at 1536 every answer had its 24
    * (`specs/52`).
    */
  val PictureSide = 1536

  /** How long a reading may take, in minutes. It is one answer for every tile,
    * and measured at 98 s for the 64 tiles of an 8192² picture and 145 s for 24
    * with another assistant: a second and a half to six seconds a tile, so from
    * eight minutes to half an hour for the 289 tiles of the largest picture
    * drift makes, 16384². The browser, the server and the call to the assistant
    * all wait as long.
    */
  val ReadingMinutes = 60

  /** The strongest a tile is repainted on the assistant's word alone: more
    * repaints shapes, which is a repair's business.
    */
  val MaxStrength = 0.6

  /** What a tile the answer left out is painted at. */
  val DefaultStrength = 0.4

  /** A tile's name on the picture: its column's letter, its row's number. */
  def nameOf(row: Int, column: Int): String =
    s"${('A' + column).toChar}${row + 1}"

  /** The tiles of a layout with their names, row by row. */
  def named(rows: List[List[ImageRegion]]): List[(String, ImageRegion)] =
    rows.zipWithIndex.flatMap((row, rowIndex) =>
      row.zipWithIndex.map((tile, column) => nameOf(rowIndex, column) -> tile)
    )

  // What the assistant writes, read leniently: a field it left out is a default
  // here, and says so in the plan's notes.
  private case class AnsweredCell(
      holds: List[String] = Nil,
      prompt: String = "",
      strength: Option[Double] = None
  )
  private case class AnsweredRepair(
      defect: String = "",
      fix: String = "",
      box: List[Double] = Nil
  )
  private case class Answer(
      kind: String = "",
      cells: Map[String, AnsweredCell] = Map.empty,
      repairs: Option[List[AnsweredRepair]] = None
  )
  private given JsonValueCodec[Answer] = JsonCodecMaker.make

  /** The last JSON object of `reply`, braces inside strings left alone — a
    * model may think aloud before it, or fence it.
    */
  def lastObject(reply: String): Option[String] = {
    def startOf(end: Int): Option[Int] = {
      // Walks back from the closing brace, counting braces outside strings; a
      // quote preceded by an odd run of backslashes is escaped.
      var depth = 0
      var inString = false
      var index = end
      var found = Option.empty[Int]
      while (index >= 0 && found.isEmpty) {
        val character = reply.charAt(index)
        if (character == '"') {
          var slashes = 0
          while (
            index - 1 - slashes >= 0 && reply.charAt(
              index - 1 - slashes
            ) == '\\'
          )
            slashes += 1
          if (slashes % 2 == 0) inString = !inString
        } else if (!inString) {
          if (character == '}') depth += 1
          else if (character == '{') {
            depth -= 1
            if (depth == 0) found = Some(index)
          }
        }
        index -= 1
      }
      found
    }
    Iterator
      .iterate(reply.lastIndexOf('}'))(end => reply.lastIndexOf('}', end - 1))
      .takeWhile(_ >= 0)
      .flatMap(end =>
        startOf(end).map(start => reply.substring(start, end + 1))
      )
      .find(candidate =>
        try { readFromString[Answer](candidate); true }
        catch { case NonFatal(_) => false }
      )
  }

  /** The plan in `reply` for the named `tiles` of a `width` × `height` picture.
    * Nothing the answer got wrong is fatal while it holds a JSON object: a tile
    * it left out keeps the default strength and no prompt of its own, a
    * strength out of range is brought back in, a repair whose box is not one is
    * dropped — and each is a note the form shows (`specs/52`).
    */
  def parse(
      reply: String,
      tiles: List[(String, ImageRegion)],
      width: Int,
      height: Int,
      seconds: Double
  ): Either[String, RedrawPlan] =
    lastObject(reply)
      .toRight("the assistant's answer holds no JSON object")
      .map { json =>
        val answer = readFromString[Answer](json)
        val missing = tiles.map(_._1).filterNot(answer.cells.contains)
        val unknown =
          answer.cells.keys.toList.filterNot(tiles.map(_._1).contains)
        var clamped = List.empty[String]
        val planned = tiles.map { (name, region) =>
          val cell = answer.cells.getOrElse(name, AnsweredCell())
          val asked = cell.strength.getOrElse(DefaultStrength)
          val strength = asked.max(0).min(MaxStrength)
          if (strength != asked) clamped :+= name
          PlannedTile(
            name,
            cell.holds.map(_.trim).filter(_.nonEmpty),
            TileSettings(region, cell.prompt.trim, strength)
          )
        }
        val (repairs, dropped) =
          answer.repairs.getOrElse(Nil).partitionMap { repair =>
            repair.box match {
              case List(left, top, right, bottom)
                  if left >= 0 && top >= 0 && right <= 1 && bottom <= 1 &&
                    left < right && top < bottom && repair.defect.trim.nonEmpty =>
                val x = math.floor(left * width).toInt
                val y = math.floor(top * height).toInt
                Left(
                  PlannedRepair(
                    repair.defect.trim,
                    repair.fix.trim,
                    ImageRegion(
                      x,
                      y,
                      (math.ceil(right * width).toInt - x).max(1),
                      (math.ceil(bottom * height).toInt - y).max(1)
                    )
                  )
                )
              case _ => Right(repair.defect.trim)
            }
          }
        def listed(names: List[String]) = names.mkString(", ")
        RedrawPlan(
          kind = answer.kind.trim,
          tiles = planned,
          repairs = repairs,
          notes = List(
            Option.when(missing.nonEmpty)(
              s"the answer left out ${listed(missing)}: those tiles keep strength $DefaultStrength and no prompt of their own"
            ),
            Option.when(unknown.nonEmpty)(
              s"the answer named tiles the picture does not have (${listed(unknown)}): ignored"
            ),
            Option.when(clamped.nonEmpty)(
              s"strength brought back within 0 and $MaxStrength on ${listed(clamped)}"
            ),
            Option.when(dropped.nonEmpty)(
              s"${dropped.size} proposed repair${
                  if (dropped.size == 1) "" else "s"
                } without a usable box dropped"
            )
          ).flatten,
          seconds = seconds
        )
      }
}

/** Has the assistant read one persisted output for its redraw (`specs/52`): one
  * call, on the whole picture scaled down with the tiles drawn on it. Refused,
  * with the reason, when no live assistant reads images or its answer cannot be
  * read.
  */
val planRedraw: PublicEndpoint[
  (String, String, RedrawPlanRequest),
  String,
  RedrawPlan,
  Any
] =
  endpoint
    .in("api")
    .post
    .in("outputs" / path[String]("date") / path[String]("file") / "redraw-plan")
    .in(jsonBody[RedrawPlanRequest])
    .out(jsonBody[RedrawPlan])
    .errorOut(stringBody)
