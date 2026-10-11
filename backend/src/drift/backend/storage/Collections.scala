package drift.backend.storage

import drift.shared.*

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec

/** A collection of stored records: the folder they are filed in under the
  * configuration directory, one file a record, and how a record is read and
  * written.
  */
final class Collection[T](val name: String)(using val codec: JsonValueCodec[T])

/** One settings document: a single record under `settings`, by its name, which
  * starts as `initial` until it is first written.
  */
final class Document[T](val name: String, val initial: T)(using
    val codec: JsonValueCodec[T]
)

/** Every collection drift stores, named once. */
object Collections {
  val architectures: Collection[Architecture] = Collection("architectures")
  val models: Collection[Model] = Collection("models")
  val runConfigurations: Collection[RunConfiguration] = Collection(
    "run-configurations"
  )
  val runtimes: Collection[Runtime] = Collection("runtimes")
  val loras: Collection[Lora] = Collection("loras")
  val projects: Collection[Project] = Collection("projects")
  val conversations: Collection[Conversation] = Collection("conversations")
  val promptTemplates: Collection[PromptTemplate] = Collection(
    "prompt-templates"
  )
  val authTokens: Collection[AuthToken] = Collection("auth-tokens")
  val upscalers: Collection[Upscaler] = Collection("upscalers")
  val pausedJobs: Collection[PausedJob] = Collection("post-process-jobs")
}
