package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** What a *build* wants and what it cannot take
  * (`specs/16-parameter-resolution.md`).
  *
  * Some flags belong to neither the architecture nor the checkpoint: flash
  * attention in the diffusion model is unavailable on the Vulkan backend,
  * offload and thread flags follow the binary and the machine. They cannot live
  * on a run configuration either — the runtime is chosen at launch, and a
  * `latest-<backend>` runtime moves under a configuration without it being
  * edited — so they are resolved when the argv is built.
  *
  * These are facts about upstream builds rather than user preferences, so they
  * ship as a backend resource and are served read-only; nothing writes them.
  */

/** A flag this build refuses, and why — the reason is shown to the user when
  * the flag is dropped, so it has to read as an explanation.
  */
case class UnsupportedFlag(flag: String, reason: String)
object UnsupportedFlag {
  given Schema[UnsupportedFlag] = Schema.derived
}

case class RuntimeRule(
    tool: RuntimeTool,
    backend: RuntimeBackend,
    /** What does the work (`specs/43`): the drift runner shares its tool and
      * backend with sd-cpp's ROCm builds, and takes other flags.
      */
    engine: RuntimeEngine,
    /** Flags this build wants unless something above it says otherwise — below
      * the run configuration, which always wins.
      */
    defaults: Map[String, String] = Map.empty,
    /** Flags dropped whoever asked for them, each with its reason. Above every
      * other layer, because the build simply cannot take them — but never
      * silently: each drop becomes a [[ResolutionNote]].
      */
    unsupported: List[UnsupportedFlag] = List.empty
)
object RuntimeRule {

  /** The rule of the runtime's tool, backend and engine, if drift ships one. */
  def forRuntime(
      runtime: Runtime,
      rules: List[RuntimeRule]
  ): Option[RuntimeRule] =
    rules.find(rule =>
      rule.tool == runtime.tool && rule.backend == runtime.backend &&
        rule.engine == runtime.engine
    )

  given Schema[RuntimeRule] = Schema.derived
  given JsonValueCodec[List[RuntimeRule]] = JsonCodecMaker.make(
    CodecMakerConfig.withDiscriminatorFieldName(None)
  )
}

/** Which layer a value came from, so a note can name it. */
enum ParameterLayer derives CanEqual {
  case Architecture, Model, Runtime, Configuration

  def label: String = this match {
    case Architecture  => "the architecture"
    case Model         => "a model"
    case Runtime       => "the runtime"
    case Configuration => "this configuration"
  }

  /** The same, short enough to sit beside a flag and its value in a chip. */
  def name: String = this match {
    case Architecture  => "architecture"
    case Model         => "model"
    case Runtime       => "runtime"
    case Configuration => "configuration"
  }
}
object ParameterLayer {
  given Schema[ParameterLayer] =
    Schema.derivedEnumeration[ParameterLayer].defaultStringBased
}

/** Something the resolution did that the command line does not show by itself.
  * A flag the user typed and cannot see in the argv must be explained, or drift
  * is lying about what it ran.
  */
enum ResolutionNote derives CanEqual {

  /** The runtime cannot take this flag, so it is not on the command line. */
  case Dropped(flag: String, from: ParameterLayer, reason: String)

  /** The runtime asked for a flag nothing above it had set. */
  case AddedByRuntime(flag: String, value: String)

  /** A layer removed a flag a lower one had set (`Model.removedParameters`,
    * `RunConfiguration.removedParameters`): the flag is not on the command
    * line, and both layers are named so the removal can be undone where it was
    * made.
    */
  case Removed(flag: String, from: ParameterLayer, by: ParameterLayer)

  def message: String = this match {
    case Dropped(flag, from, reason) =>
      s"$flag — set by ${from.label}, dropped: $reason"
    case Removed(flag, from, by) =>
      s"$flag — set by ${from.label}, removed by ${by.label}"
    case AddedByRuntime(flag, value) =>
      val shown = if (value.isEmpty) flag else s"$flag $value"
      s"$shown — added by the runtime"
  }
}
object ResolutionNote {
  given Schema[ResolutionNote] = Schema.derived
}

/** One flag as the layers leave it: the value that would be passed, and which
  * layer asked for it.
  *
  * The layer is half the answer, not decoration: a model that sets
  * `--attn-scale` to the same number its architecture already had looks
  * identical to no override at all unless the form says where the value came
  * from (François, 2026-09-20).
  */
case class ResolvedParameter(
    flag: String,
    value: String,
    layer: ParameterLayer
)

/** The parameters a launch will actually pass, and everything worth saying
  * about how they were arrived at.
  */
case class ParameterResolution(
    arguments: List[(String, String)],
    notes: List[ResolutionNote]
)

private val runtimeRulesBase = endpoint.in("api")

/** Every rule drift ships. Read-only: these describe upstream builds, not the
  * user's choices.
  */
val listRuntimeRules: PublicEndpoint[Unit, Unit, List[RuntimeRule], Any] =
  runtimeRulesBase.get.in("runtime-rules").out(jsonBody[List[RuntimeRule]])
