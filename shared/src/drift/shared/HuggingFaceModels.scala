package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** What a repository's model makes (`specs/36-huggingface-examples.md`): an
  * image or video of its card's gallery (the card's `widget`), with the prompt
  * the card gives for it, or a media file found in the repository.
  */
case class ModelExample(
    url: String,
    video: Boolean = false,
    prompt: Option[String] = None
)

case class HuggingFaceModelInfo(
    id: String,
    pipeline_tag: Option[String] = None,
    library_name: Option[String] = None,
    tags: List[String] = List.empty,
    downloads: Option[Long] = None,
    likes: Option[Long] = None,
    last_modified: Option[String] = None,
    author: Option[String] = None,
    /** Filled by drift, not HuggingFace: the first few examples, for the
      * result's tile — the next is tried when one does not load — and how many
      * there are.
      */
    previews: List[ModelExample] = Nil,
    exampleCount: Int = 0
)
object HuggingFaceModelInfo {
  given JsonValueCodec[List[HuggingFaceModelInfo]] = JsonCodecMaker.make
}

/** The `lfs` block HuggingFace attaches to large files. Only present when the
  * listing is requested with `?blobs=true`.
  */
case class HuggingFaceLfs(
    sha256: Option[String] = None,
    size: Option[Long] = None
)

case class HuggingFaceFileInfo(
    rfilename: String,
    size: Option[Long] = None,
    lfs: Option[HuggingFaceLfs] = None,
    /** The git blob id, which is the file's ETag — and so its blob name in the
      * HuggingFace cache — when the file is not in LFS.
      */
    blobId: Option[String] = None
) {

  /** The content SHA256, which is also the blob filename in HuggingFace's own
    * cache — simultaneously the verification hash and the cache key.
    */
  def sha256: Option[String] = lfs.flatMap(_.sha256).map(_.toLowerCase)
}

/** HuggingFace's `gated` flag: absent for an open repository, else the kind of
  * gate — "auto", where accepting the terms on the site is enough, or "manual",
  * where the author approves each request by hand. Either way a download needs
  * a token from an account that has been let in, so the browser says so before
  * a file is picked rather than leaving it to a 403.
  *
  * The field is a JSON union — `false`, or a string — which no derived codec
  * reads, hence this one; it writes the same shape back.
  */
case class Gated(kind: Option[String] = None) {
  def isGated: Boolean = kind.isDefined

  /** Whether the author lets people in one by one. */
  def isManual: Boolean = kind.contains("manual")
}
object Gated {
  given Schema[Gated] = Schema.derived

  given JsonValueCodec[Gated] = new JsonValueCodec[Gated] {
    def nullValue: Gated = Gated()

    def decodeValue(in: JsonReader, default: Gated): Gated = {
      val token = in.nextToken()
      in.rollbackToken()
      if (token == '"') Gated(Some(in.readString(null)))
      else if (token == 'n') in.readNullOrError(default, "expected null")
      else {
        in.readBoolean()
        Gated()
      }
    }

    def encodeValue(x: Gated, out: JsonWriter): Unit =
      x.kind match {
        case Some(kind) => out.writeVal(kind)
        case None       => out.writeVal(false)
      }
  }
}

case class HuggingFaceModelDetail(
    id: String,
    siblings: List[HuggingFaceFileInfo],
    /** The commit of `main`, which names the snapshot directory in the cache.
      */
    sha: Option[String] = None,
    /** Whether HuggingFace gates this repository, and how. */
    gated: Gated = Gated(),
    /** Filled by drift for the browser's Examples tab: the card's gallery, then
      * the repository's other media files.
      */
    examples: List[ModelExample] = Nil
)
object HuggingFaceModelDetail {
  given JsonValueCodec[HuggingFaceModelDetail] = JsonCodecMaker.make
  given JsonValueCodec[Option[HuggingFaceModelDetail]] = JsonCodecMaker.make
}

/** A repository's model card, rendered to HTML and cleaned by the backend
  * (`specs/24`), so the page inserts it as-is. Without `html` there is nothing
  * to show: `gated` when the repository wants its terms accepted and a token,
  * otherwise it has no README.
  */
case class HuggingFaceModelCard(
    html: Option[String] = None,
    gated: Boolean = false
)
object HuggingFaceModelCard {
  given JsonValueCodec[HuggingFaceModelCard] = JsonCodecMaker.make
}

/** `baseModel` narrows the search to that repository's adapters — LoRAs trained
  * on it (`specs/33-lora-sources.md`).
  */
lazy val searchHuggingFaceModels: PublicEndpoint[
  (
      String,
      Option[Int],
      Option[Int],
      Option[String],
      Option[Int],
      Option[
        String
      ]
  ),
  String,
  List[HuggingFaceModelInfo],
  Any
] =
  base.get
    .in("hf-search")
    .errorOut(upstreamFailure)
    .in(query[String]("q"))
    .in(query[Option[Int]]("limit"))
    .in(query[Option[Int]]("offset"))
    .in(query[Option[String]]("sort"))
    .in(query[Option[Int]]("direction"))
    .in(query[Option[String]]("baseModel"))
    .out(jsonBody[List[HuggingFaceModelInfo]])

lazy val getHuggingFaceModelDetail
    : PublicEndpoint[(String, String), String, Option[
      HuggingFaceModelDetail
    ], Any] =
  base.get
    .in("hf-model" / path[String] / path[String])
    .errorOut(upstreamFailure)
    .out(jsonBody[Option[HuggingFaceModelDetail]])

/** The repository's README, rendered and cleaned (`specs/24`). */
lazy val getHuggingFaceModelCard
    : PublicEndpoint[(String, String), String, HuggingFaceModelCard, Any] =
  base.get
    .in("hf-card" / path[String] / path[String])
    .errorOut(upstreamFailure)
    .out(jsonBody[HuggingFaceModelCard])
