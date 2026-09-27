package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** Where a model or a LoRA came from, as a mark beside its name (François,
  * 2026-09-17), and on the browser's source row. The sites' own logos are
  * images drift does not carry; their initials in their colour are recognised
  * just as well in the space a list row can spare, and say the rest in the
  * tooltip.
  */
object ProviderIcon {

  private def initials(kind: ModelSourceType): String = kind match {
    case ModelSourceType.Civitai     => "C"
    case ModelSourceType.HuggingFace => "HF"
    case ModelSourceType.ModelScope  => "MS"
    case ModelSourceType.Local       => "💾"
  }

  private def colourOf(kind: ModelSourceType): String = kind match {
    case ModelSourceType.Civitai     => "is-civitai"
    case ModelSourceType.HuggingFace => "is-huggingface"
    case ModelSourceType.ModelScope  => "is-modelscope"
    case ModelSourceType.Local       => "is-local"
  }

  /** The site's mark alone, its name in the tooltip. */
  def of(kind: ModelSourceType, mods: Mod[HtmlElement]*): HtmlElement =
    span(
      cls := s"provider-icon ${colourOf(kind)}",
      title := SourceBrowser.name(kind),
      initials(kind),
      mods
    )

  /** The mark of one file's source, its whole origin in the tooltip. */
  def of(source: ModelSource): HtmlElement =
    of(source.kind, title := source.lines.mkString(" · "))

  /** The mark of a LoRA, whose files come from one place; a LoRA with no file
    * yet has nothing to show.
    */
  def of(lora: Lora): Option[HtmlElement] =
    lora.files.headOption.map(file => of(file.source))
}
