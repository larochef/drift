package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

enum ModelSourceType derives CanEqual {
  case HuggingFace, ModelScope, Civitai, Local
}

sealed trait ModelSource derives CanEqual {

  /** Which kind it is, for a form that shows the source it was given. */
  def kind: ModelSourceType = this match {
    case _: HuggingFace => ModelSourceType.HuggingFace
    case _: ModelScope  => ModelSourceType.ModelScope
    case _: Civitai     => ModelSourceType.Civitai
    case _: Local       => ModelSourceType.Local
  }

  /** Where the file comes from, as the model list and the model form show it:
    * the site and what it takes to find it again, a line each — a HuggingFace
    * or ModelScope file name is long enough to want its own.
    */
  def lines: List[String] = this match {
    case HuggingFace(repo, file) => List(s"HuggingFace $repo", file)
    case ModelScope(repo, file)  => List(s"ModelScope $repo", file)
    case Civitai(id, _, _, file) => List(s"Civitai model $id", file)
    case Local(path)             => List("On this machine", path)
  }

  /** The file's own name, wherever it comes from. */
  def fileName: String = this match {
    case HuggingFace(_, file) => file.split('/').last
    case ModelScope(_, file)  => file.split('/').last
    case Civitai(_, _, _, f)  => f
    case Local(path)          => path.split('/').last
  }
}
object ModelSource {

  /** Hierarchies are discriminated by a lowercase `"type"`, and the key may sit
    * anywhere in the object -- jsoniter otherwise demands it come first, which
    * a hand-edited config file has no reason to respect.
    */
  given JsonValueCodec[ModelSource] = JsonCodecMaker.make(
    CodecMakerConfig
      .withAdtLeafClassNameMapper(n =>
        JsonCodecMaker.simpleClassName(n).toLowerCase
      )
      .withRequireDiscriminatorFirst(false)
  )
}

case class HuggingFace(repo: String, filename: String) extends ModelSource

/** A file of a ModelScope repository (`specs/37-modelscope.md`), `repo` being
  * `owner/name` and `filename` its path in the repository. Fetched into drift's
  * own `modelscope/` tree, not a ModelScope SDK cache.
  */
case class ModelScope(repo: String, filename: String) extends ModelSource

/** A split safetensors model is named by its `*.safetensors.index.json`, which
  * maps every tensor to one of the shard files beside it. sd-cpp loads the
  * index; drift downloads and checks the shards with it
  * (`specs/34-sharded-safetensors.md`).
  */
object ShardedSafetensors {
  def isIndex(filename: String): Boolean =
    filename.toLowerCase.endsWith(".safetensors.index.json")
}

/** Civitai downloads are addressed by `modelId` + `fileId`; `filename` is what
  * the file is called on disk. `fileId` is required and undefaulted on purpose
  * -- a Civitai source without one cannot be fetched, so that should not
  * compile.
  */
case class Civitai(
    modelId: String,
    versionId: String,
    fileId: String,
    filename: String
) extends ModelSource
case class Local(path: String) extends ModelSource

case class Model(
    id: String,
    familyId: String,
    label: String,
    source: ModelSource,
    format: String,
    parameters: Map[String, String],
    /** Flags this checkpoint does not want at all, whatever the architecture
      * below it set (`specs/16-parameter-resolution.md`). A value cannot say
      * this: `""` is the convention for a valueless flag that *is* passed, so
      * an "empty" override would put `--steps ""` on the command line. A layer
      * above can set the flag again.
      */
    removedParameters: List[String],
    builtIn: Boolean
)
object Model {
  given JsonValueCodec[Model] = JsonCodecMaker.make(storedRecord)
  given JsonValueCodec[List[Model]] = JsonCodecMaker.make(storedRecord)
  given JsonValueCodec[Option[Model]] = JsonCodecMaker.make(storedRecord)
}

lazy val listModels: PublicEndpoint[Unit, Unit, List[Model], Any] =
  base.get.in("models").out(jsonBody[List[Model]])

lazy val getModel: PublicEndpoint[String, Unit, Option[Model], Any] =
  base.get.in("models" / path[String]).out(jsonBody[Option[Model]])

lazy val createModel: PublicEndpoint[Model, Unit, Model, Any] =
  base.post.in("models").in(jsonBody[Model]).out(jsonBody[Model])

lazy val updateModel
    : PublicEndpoint[(String, Model), Unit, Option[Model], Any] =
  base.put
    .in("models" / path[String])
    .in(jsonBody[Model])
    .out(jsonBody[Option[Model]])

lazy val deleteModel: PublicEndpoint[String, Unit, Boolean, Any] =
  base.delete.in("models" / path[String]).out(jsonBody[Boolean])
