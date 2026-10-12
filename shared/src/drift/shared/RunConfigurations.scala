package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

case class RunConfiguration(
    id: String,
    label: String,
    architectureId: String,
    assignments: Map[String, String],
    overriddenParameters: Map[String, String],
    /** Flags this configuration does not want at all, whatever the
      * architecture, its models or the runtime set
      * (`specs/16-parameter-resolution.md`). The top layer removes as it
      * overrides: nothing above it can put the flag back.
      */
    removedParameters: List[String],
    createdAt: Long,
    /** When this configuration last launched a session — `createdAt` until the
      * first launch, so a fresh configuration sorts like a just-used one. The
      * inference page orders its grid by this, most recent first.
      */
    lastUsedAt: Long,
    /** LoRAs applied wherever this configuration runs — its generation form
      * starts with them, a tiled job sends them with every tile
      * (`specs/28-configuration-loras.md`). Only LoRAs of its architecture.
      */
    loras: List[ConfiguredLora],
    /** The assistant system template this configuration prefers over its
      * architecture's default (`specs/32-prompt-library.md`); none means the
      * architecture's. A project's assistant is offered a switch to it when the
      * configuration's model starts.
      */
    assistantTemplateId: Option[String],
    /** The engine it runs on, one of its architecture's runners (`specs/43`):
      * every launch path resolves its runtime from it.
      */
    runner: RuntimeEngine
)
object RunConfiguration {
  // No discriminator: `RuntimeEngine` encodes as a plain string ("SdCpp").
  given JsonValueCodec[RunConfiguration] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[RunConfiguration]] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[RunConfiguration]] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
}

lazy val listRunConfigurations
    : PublicEndpoint[Unit, Unit, List[RunConfiguration], Any] =
  base.get.in("run-configurations").out(jsonBody[List[RunConfiguration]])

lazy val getRunConfiguration
    : PublicEndpoint[String, Unit, Option[RunConfiguration], Any] =
  base.get
    .in("run-configurations" / path[String])
    .out(jsonBody[Option[RunConfiguration]])

lazy val createRunConfiguration
    : PublicEndpoint[RunConfiguration, Unit, RunConfiguration, Any] =
  base.post
    .in("run-configurations")
    .in(jsonBody[RunConfiguration])
    .out(jsonBody[RunConfiguration])

lazy val updateRunConfiguration
    : PublicEndpoint[(String, RunConfiguration), Unit, Option[
      RunConfiguration
    ], Any] =
  base.put
    .in("run-configurations" / path[String])
    .in(jsonBody[RunConfiguration])
    .out(jsonBody[Option[RunConfiguration]])

lazy val deleteRunConfiguration: PublicEndpoint[String, Unit, Boolean, Any] =
  base.delete.in("run-configurations" / path[String]).out(jsonBody[Boolean])
