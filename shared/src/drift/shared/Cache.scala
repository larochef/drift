package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Whether a registered model's weights are on disk.
  *
  * `Missing` and `Broken` are distinct on purpose: a missing file can be
  * downloaded, a broken `Local` path cannot — the user pointed at something
  * that is not there.
  */
enum CacheState derives CanEqual {
  case Cached, Missing, Broken
}

case class ModelCacheStatus(
    modelId: String,
    state: CacheState,
    path: Option[String] = None,
    bytes: Option[Long] = None
)
object ModelCacheStatus {
  // No discriminator: `CacheState` has only singleton cases, so jsoniter encodes
  // it as a plain string ("Cached") rather than an object ({"type":"Cached"}).
  // The config has to be inline -- the macro needs a constant expression.
  given JsonValueCodec[ModelCacheStatus] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[ModelCacheStatus]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** Where a cached file came from — the on-disk view tabs by this. `Civitai`
  * means a drift-root file whose sidecar names Civitai as the source;
  * everything else in the drift models directory is `Local`. `Lora` is reserved
  * for `specs/09-lora-management.md`; nothing produces it yet.
  */
enum CachedFileKind derives CanEqual {
  case HuggingFace, ModelScope, Civitai, Local, Lora
}
object CachedFileKind {
  // String-encoded like the other singleton enums; the explicit schema keeps
  // tapir's derivation in agreement (see RuntimeBackend).
  given Schema[CachedFileKind] =
    Schema.derivedEnumeration[CachedFileKind].defaultStringBased
}

/** One file the cache actually holds, whether or not a model references it. The
  * on-disk view of the Model Cache page is built from these; orphans —
  * `referencedBy` empty — are exactly what that view exists to surface.
  */
case class CachedFileEntry(
    /** Real path; also the deletion key. */
    path: String,
    bytes: Long,
    /** Which store holds it: "drift" or "huggingface". */
    root: String,
    /** Family directory (drift) or repository name (HuggingFace). */
    group: String,
    /** Something readable: snapshot filename, or the sidecar's model name. */
    label: String,
    kind: CachedFileKind,
    referencedBy: List[String] = List.empty,
    /** A `.part` / `.incomplete` left by an interrupted download. */
    partial: Boolean = false,
    /** Where the file came from, as a model adopting it should reference it — a
      * HuggingFace repo file or a Civitai file — set only when that source
      * resolves back to this very file. None means referencing it in place, by
      * its path.
      */
    source: Option[ModelSource] = None
)
object CachedFileEntry {
  // No discriminator, so `CachedFileKind` encodes as a plain string. The
  // config must be inline -- the macro needs a constant expression.
  given JsonValueCodec[CachedFileEntry] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[CachedFileEntry]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

case class FileEntry(
    name: String,
    path: String,
    isDirectory: Boolean,
    size: Long,
    extension: String
)
object FileEntry {
  given JsonValueCodec[List[FileEntry]] = JsonCodecMaker.make
}

/** Cache state for every registered model. Cheap enough to return wholesale: it
  * is a stat per model, and the UI wants the whole picture on every page it
  * appears on.
  */
lazy val cacheStatus: PublicEndpoint[Unit, Unit, List[ModelCacheStatus], Any] =
  base.get.in("cache" / "status").out(jsonBody[List[ModelCacheStatus]])

lazy val listCachedFiles
    : PublicEndpoint[Unit, Unit, List[CachedFileEntry], Any] =
  base.get.in("cache" / "files").out(jsonBody[List[CachedFileEntry]])

/** Deletes one cached file by its listed path. The backend refuses anything
  * outside its roots, and a file it could not delete, with the reason in words;
  * whether deleting is *wise* — a ready run configuration may depend on the
  * file — is the frontend's warning to give.
  */
lazy val deleteCachedFile: PublicEndpoint[String, String, Unit, Any] =
  base.delete
    .in("cache" / "files")
    .in(query[String]("path"))
    .errorOut(stringBody)

/** The Civitai file ids of one model already present in the cache — what the
  * browser's "downloaded" markers are built from. Read off the on-disk
  * `<fileId>-<filename>` naming, so it needs no registry entry to answer.
  */
lazy val listCivitaiCachedFileIds
    : PublicEndpoint[String, Unit, List[String], Any] =
  base.get
    .in("cache" / "civitai-files")
    .in(query[String]("modelId"))
    .out(jsonBody[List[String]])

lazy val getHomeDirectory: PublicEndpoint[Unit, Unit, String, Any] =
  base.get.in("files" / "home").out(stringBody)

lazy val listDirectory: PublicEndpoint[String, Unit, List[FileEntry], Any] =
  base.get
    .in("files" / "list")
    .in(query[String]("path"))
    .out(jsonBody[List[FileEntry]])
