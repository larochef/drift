package drift.backend.routes

import drift.shared.ModelExample

import org.commonmark.Extension
import org.commonmark.ext.autolink.AutolinkExtension
import org.commonmark.ext.gfm.tables.TablesExtension
import org.commonmark.parser.Parser
import org.commonmark.renderer.html.HtmlRenderer
import org.jsoup.Jsoup
import org.jsoup.nodes.Document
import org.jsoup.safety.Safelist

/** Model descriptions as the browsers show them (`specs/24`): a HuggingFace
  * model card is Markdown with inline HTML, a Civitai description is HTML, and
  * both are written by strangers. They are rendered and cleaned here, so the
  * page inserts the result as-is and needs no Markdown library — nothing from
  * either site reaches the DOM uncleaned.
  */
object ModelDescriptions {

  private val extensions = java.util.List.of[Extension](
    TablesExtension.create(),
    AutolinkExtension.create()
  )
  private val parser = Parser.builder().extensions(extensions).build()
  private val renderer = HtmlRenderer.builder().extensions(extensions).build()

  /** A card's YAML front matter (after an optional byte-order mark) is metadata
    * for the Hub, not prose.
    */
  private val FrontMatter =
    """(?s)\A\x{FEFF}?---\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\z)""".r

  /** jsoup's relaxed list — text, lists, tables, links, images — plus what
    * cards use for layout and demos: `align`, rules, strike-through, `details`
    * and videos. Relative URLs resolve against the base given to `clean` and
    * anything that does not end up http(s) is dropped; links open outside
    * drift.
    */
  private val safelist: Safelist =
    Safelist
      .relaxed()
      .addTags(
        "hr",
        "del",
        "s",
        "details",
        "summary",
        "video",
        "source",
        "figure",
        "figcaption"
      )
      .addAttributes(":all", "align")
      .addAttributes(
        "video",
        "src",
        "poster",
        "controls",
        "loop",
        "muted",
        "playsinline",
        "width",
        "height"
      )
      .addAttributes("source", "src", "type")
      .addProtocols("video", "src", "http", "https")
      .addProtocols("video", "poster", "http", "https")
      .addProtocols("source", "src", "http", "https")
      .addEnforcedAttribute("a", "target", "_blank")
      .addEnforcedAttribute("a", "rel", "noopener noreferrer")

  private val GalleryTag = """(?i)<gallery\s*/?>(?:\s*</gallery>)?""".r

  /** Whether a README asks HuggingFace for its widget gallery. */
  def hasGallery(markdown: String): Boolean =
    GalleryTag.findFirstIn(markdown).isDefined

  /** `<Gallery />` replaced by the card's examples, as HuggingFace renders it
    * (`specs/36-huggingface-examples.md`): a figure per image or video, its
    * prompt as the caption. Without examples the tag goes, as the cleaner would
    * drop it anyway.
    */
  def withGallery(
      markdown: String,
      examples: List[ModelExample]
  ): String = {
    def escaped(text: String): String =
      text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
    val figures = examples.map { example =>
      val media =
        if (example.video)
          s"""<video src="${escaped(
              example.url
            )}" controls muted loop playsinline></video>"""
        else
          s"""<img src="${escaped(example.url)}" alt="${escaped(
              example.prompt.getOrElse("")
            )}">"""
      val caption =
        example.prompt
          .map(p => s"<figcaption>${escaped(p)}</figcaption>")
          .getOrElse("")
      s"<figure>$media$caption</figure>"
    }
    // A raw HTML block needs blank lines around it to stay one in CommonMark.
    val block =
      if (figures.isEmpty) ""
      else figures.mkString("\n\n<div>\n", "\n", "\n</div>\n\n")
    GalleryTag.replaceAllIn(
      markdown,
      scala.util.matching.Regex.quoteReplacement(block)
    )
  }

  /** Markdown rendered to HTML, then cleaned like any HTML. */
  def fromMarkdown(markdown: String, baseUri: String): String =
    fromHtml(
      renderer.render(parser.parse(FrontMatter.replaceFirstIn(markdown, ""))),
      baseUri
    )

  /** An assistant's reply (`specs/20`): Markdown like a card's, cleaned the
    * same way, but with no front matter to strip — a reply opening on a rule
    * is prose — and no base, so a relative link is dropped.
    */
  def fromReply(markdown: String): String =
    fromHtml(renderer.render(parser.parse(markdown)), "")

  /** HTML reduced to the safelist, relative URLs made absolute against
    * `baseUri`.
    */
  def fromHtml(html: String, baseUri: String): String =
    Jsoup.clean(
      html,
      baseUri,
      safelist,
      Document.OutputSettings().prettyPrint(false)
    )
}
