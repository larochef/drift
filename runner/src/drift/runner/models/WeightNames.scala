package drift.runner.models

import drift.runner.formats.FormatException

/** The names the loaders read, from the names a file stores (sd-cpp's
  * `name_conversion.cpp`, `specs/42`): one map at load, so that every loader
  * reads one naming whoever exported the file.
  *   - ComfyUI's and the single files' prefixes are dropped:
  *     `model.diffusion_model.` (a Civitai checkpoint, sd-cpp's GGUFs),
  *     `diffusion_model.`, `net.` (PiD's), `first_stage_model.` and `vae.` (a
  *     VAE saved out of a whole checkpoint).
  *   - A diffusers `AutoencoderKL` (FLUX.1's, FLUX.2's `AutoencoderKLFlux2`:
  *     `encoder.down_blocks.N.resnets.N`, `mid_block.attentions.0.to_q`) gets
  *     the original LDM names (`encoder.down.N.block.N`, `mid.attn_1.q`), the
  *     decoder's levels counted from the other end, as `FluxVae` reads them.
  */
object WeightNames {

  val Prefixes: Seq[String] = Seq(
    "model.diffusion_model.",
    "diffusion_model.",
    "net.",
    "first_stage_model.",
    "vae."
  )

  /** The loaders' name of every stored one, as `loaded name → stored name`. Two
    * stored names that come to the same one (a file holding a tensor both with
    * and without a prefix) are refused by name.
    */
  def canonical(stored: Iterable[String]): Map[String, String] = {
    val stripped = stored.map(name => withoutPrefix(name) -> name).toSeq
    val rename: String => String =
      if (isDiffusersAutoencoder(stripped.map(_._1))) {
        val levels = decoderLevels(stripped.map(_._1))
        name => ldmVaeName(name, levels)
      } else identity
    val renamed = stripped.map((name, original) => rename(name) -> original)
    renamed.groupBy(_._1).find(_._2.size > 1).foreach { (name, clashing) =>
      throw new FormatException(
        s"${clashing.map(_._2).sorted.mkString(" and ")} are both $name"
      )
    }
    renamed.toMap
  }

  def withoutPrefix(name: String): String =
    Prefixes.find(name.startsWith).fold(name)(name.stripPrefix)

  /** diffusers' AutoencoderKL, by its middle attention's own names (Wan's
    * diffusers VAE fuses `to_qkv`, and is not this one).
    */
  private def isDiffusersAutoencoder(names: Seq[String]): Boolean =
    names.exists(name =>
      name.matches(
        "(encoder|decoder)\\.mid_block\\.attentions\\.0\\.(to_q|query)\\..*"
      )
    )

  private val UpBlock = "decoder\\.up_blocks\\.(\\d+)\\..*".r

  private def decoderLevels(names: Seq[String]): Int =
    names
      .collect { case UpBlock(level) => level.toInt + 1 }
      .maxOption
      .getOrElse(0)

  private val Renames: Seq[(String, String)] = Seq(
    "^quant_conv\\." -> "encoder.quant_conv.",
    "^post_quant_conv\\." -> "decoder.post_quant_conv.",
    "^(encoder|decoder)\\.conv_norm_out\\." -> "$1.norm_out.",
    "^encoder\\.down_blocks\\.(\\d+)\\.resnets\\." -> "encoder.down.$1.block.",
    "^encoder\\.down_blocks\\.(\\d+)\\.downsamplers\\.0\\." -> "encoder.down.$1.downsample.",
    "\\.mid_block\\.attentions\\.0\\." -> ".mid.attn_1.",
    "\\.(group_norm)\\." -> ".norm.",
    "\\.(to_q|query)\\." -> ".q.",
    "\\.(to_k|key)\\." -> ".k.",
    "\\.(to_v|value)\\." -> ".v.",
    "\\.(to_out\\.0|proj_attn)\\." -> ".proj_out.",
    "\\.conv_shortcut\\." -> ".nin_shortcut."
  )

  private val MidResidual = "(.*)\\.mid_block\\.resnets\\.(\\d+)\\.(.*)".r
  private val UpLevel =
    "decoder\\.up_blocks\\.(\\d+)\\.(resnets|upsamplers\\.0)\\.(.*)".r

  /** One diffusers AutoencoderKL name as the LDM checkpoints hold it; the
    * decoder has `levels` levels.
    */
  def ldmVaeName(name: String, levels: Int): String = {
    val placed = name match {
      case MidResidual(side, index, rest) =>
        s"$side.mid.block_${index.toInt + 1}.$rest"
      case UpLevel(level, "resnets", rest) =>
        s"decoder.up.${levels - 1 - level.toInt}.block.$rest"
      case UpLevel(level, _, rest) =>
        s"decoder.up.${levels - 1 - level.toInt}.upsample.$rest"
      case other => other
    }
    Renames.foldLeft(placed)((renamed, rename) =>
      renamed.replaceAll(rename._1, rename._2)
    )
  }
}
