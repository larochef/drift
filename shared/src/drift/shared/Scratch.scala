package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Free play's write side (`specs/22-free-play-and-scratch-generations.md`).
  *
  * A generation submitted with `scratch=true` lands under `outputs/scratch/`
  * and writes no sidecar, so history — which only counts date directories —
  * never sees it. **Keep** promotes one into the outputs proper: the files move
  * into their day, the sidecar is written, and with a project named the recipe
  * becomes a version of it like any submission would.
  */

/** Where a kept generation should land. Without a project it is a plain gallery
  * entry; with one it goes through the same versioning as a submission.
  */
case class KeepRequest(projectId: Option[String] = None)
object KeepRequest {
  given Schema[KeepRequest] = Schema.derived
  given JsonValueCodec[KeepRequest] = JsonCodecMaker.make(
    CodecMakerConfig.withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
}

private val scratchBase = endpoint.in("api")

/** Promotes one scratch generation to a kept one. Answers `None` when the id is
  * not a live scratch generation — it was already kept, or the session that
  * made it is gone, which takes its files with it.
  */
val keepScratchGeneration
    : PublicEndpoint[(String, KeepRequest), Unit, Option[Generation], Any] =
  scratchBase.post
    .in("scratch" / path[String]("generationId") / "keep")
    .in(jsonBody[KeepRequest])
    .out(jsonBody[Option[Generation]])

/** Throws away everything in `outputs/scratch/`, answering with the ids that
  * went. Runs by itself when a session stops and at startup; this is the
  * button.
  */
val clearScratch: PublicEndpoint[Unit, Unit, List[String], Any] =
  scratchBase.delete.in("scratch").out(jsonBody[List[String]])
