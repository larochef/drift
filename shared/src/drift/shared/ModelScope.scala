package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** One repository of a ModelScope search (`specs/37-modelscope.md`), in drift's
  * words: ModelScope's own answer is read by the backend and never reaches the
  * page.
  */
case class ModelScopeModelInfo(
    /** `owner/name`. */
    id: String,
    downloads: Long = 0,
    stars: Long = 0,
    /** ModelScope files it as a LoRA (its AIGC type). */
    lora: Boolean = false,
    baseModels: List[String] = Nil,
    /** Seconds since the epoch. */
    updatedAt: Option[Long] = None,
    /** The first few examples, for the result's tile — the next is tried when
      * one does not load — and how many there are.
      */
    previews: List[ModelExample] = Nil,
    exampleCount: Int = 0
)

/** A page of results and how many the search matches in all. */
case class ModelScopeSearchPage(
    models: List[ModelScopeModelInfo] = Nil,
    total: Int = 0
)
object ModelScopeSearchPage {
  given JsonValueCodec[ModelScopeSearchPage] = JsonCodecMaker.make
}

/** A file of a repository, with the hash ModelScope lists for every file. */
case class ModelScopeFileInfo(
    path: String,
    size: Option[Long] = None,
    sha256: Option[String] = None
)

/** An opened repository: its files, and what its model makes — the cover images
  * of its versions, with their prompts, then its media files.
  */
case class ModelScopeModelDetail(
    id: String,
    files: List[ModelScopeFileInfo] = Nil,
    examples: List[ModelExample] = Nil
)
object ModelScopeModelDetail {
  given JsonValueCodec[Option[ModelScopeModelDetail]] = JsonCodecMaker.make
}

/** The repository's README, rendered and cleaned; none when it has none. */
case class ModelScopeModelCard(html: Option[String] = None)
object ModelScopeModelCard {
  given JsonValueCodec[ModelScopeModelCard] = JsonCodecMaker.make
}

/** ModelScope's search. `sort` is one of `ModelScopeSorts`; every `baseModel`
  * given widens the base-model filter (ModelScope ORs them); `lorasOnly` keeps
  * the repositories ModelScope files as LoRAs.
  */
val searchModelScope: PublicEndpoint[
  (String, Int, Option[String], List[String], Boolean),
  String,
  ModelScopeSearchPage,
  Any
] =
  base.get
    .in("ms-search")
    .in(query[String]("q"))
    .in(query[Int]("page"))
    .in(query[Option[String]]("sort"))
    .in(query[List[String]]("baseModel"))
    .in(query[Boolean]("lorasOnly"))
    .errorOut(upstreamFailure)
    .out(jsonBody[ModelScopeSearchPage])

val getModelScopeModelDetail: PublicEndpoint[
  (String, String),
  String,
  Option[ModelScopeModelDetail],
  Any
] =
  base.get
    .in("ms-model" / path[String] / path[String])
    .errorOut(upstreamFailure)
    .out(jsonBody[Option[ModelScopeModelDetail]])

val getModelScopeModelCard
    : PublicEndpoint[(String, String), String, ModelScopeModelCard, Any] =
  base.get
    .in("ms-card" / path[String] / path[String])
    .errorOut(upstreamFailure)
    .out(jsonBody[ModelScopeModelCard])

/** The orders a ModelScope search offers: its key, then the words for it. */
val ModelScopeSorts: List[(String, String)] = List(
  "DownloadsCount" -> "Most downloaded",
  "StarsCount" -> "Most stars",
  "GmtModified" -> "Recently updated",
  "GmtCreated" -> "Recently created",
  "Default" -> "ModelScope's order"
)
