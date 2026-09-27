package drift.runner.models

import drift.runner.formats.{FormatException, ModelConfig}
import drift.runner.ops.*

/** Keys whose place in `config.json` changed across transformers versions, and
  * the shapes several families read alike.
  */
object HuggingFaceConfigs {

  /** The Qwen hybrids' block shapes from a text config. */
  def hybridShape(config: ModelConfig, deltaGate: Activation): HybridShape = {
    val rope =
      if (config.has("rope_parameters")) Some(config.section("rope_parameters"))
      else None
    val partial = rope
      .filter(_.has("partial_rotary_factor"))
      .map(_.double("partial_rotary_factor"))
      .getOrElse(config.double("partial_rotary_factor"))
    val headDimension = config.int("head_dim")
    val keyDimension = config.int("linear_key_head_dim")
    if (keyDimension != config.int("linear_value_head_dim"))
      throw new FormatException(
        s"${config.source}: key and value heads of different sizes"
      )
    HybridShape(
      hidden = config.int("hidden_size"),
      heads = config.int("num_attention_heads"),
      kvHeads = config.int("num_key_value_heads"),
      headDimension = headDimension,
      rotaryDimensions = (headDimension * partial).toInt,
      ropeTheta = ropeTheta(config).toFloat,
      ropeSections = ropeSections(config),
      rmsEpsilon = config.double("rms_norm_eps").toFloat,
      deltaRule = DeltaRule(
        config.int("linear_num_key_heads"),
        config.int("linear_num_value_heads"),
        keyDimension,
        tiledHeads = false
      ),
      convTaps = config.int("linear_conv_kernel_dim"),
      feedForward =
        if (config.has("num_experts"))
          RoutedExperts(
            config.int("num_experts"),
            config.int("num_experts_per_tok"),
            config.int("moe_intermediate_size"),
            config.int("shared_expert_intermediate_size")
          )
        else DenseFeedForward(config.int("intermediate_size")),
      normOffset = 1f,
      deltaGate = deltaGate
    )
  }

  /** mRoPE's sections (`mrope_section` in `rope_parameters`, or in
    * `rope_scaling` before transformers 5), interleaved or contiguous; plain
    * RoPE without them.
    */
  def ropeSections(config: ModelConfig): RopeSections =
    Seq("rope_parameters", "rope_scaling")
      .filter(config.has)
      .map(config.section)
      .find(_.has("mrope_section"))
      .map { parameters =>
        val Seq(t, h, w) = parameters.ints("mrope_section").take(3)
        if (
          parameters.has("mrope_interleaved") &&
          parameters.boolean("mrope_interleaved")
        ) RopeSections.Interleaved(t, h, w)
        else RopeSections.Contiguous(t, h, w)
      }
      .getOrElse(RopeSections.Single)

  /** transformers 5 moved `rope_theta` into `rope_parameters` (Qwen 3.8 has it
    * there too); older checkpoints keep it at the top.
    */
  def ropeTheta(config: ModelConfig): Double =
    if (
      config.has("rope_parameters") && config
        .section("rope_parameters")
        .has("rope_theta")
    )
      config.section("rope_parameters").double("rope_theta")
    else if (config.has("rope_theta")) config.double("rope_theta")
    else throw new FormatException(s"${config.source} has no rope_theta")
}
