package drift.backend.routes

import drift.shared.ModelExample

import java.net.{URI, URLDecoder}
import java.nio.charset.StandardCharsets
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker

/** Where a HuggingFace repository shows what its model makes
  * (`specs/36-huggingface-examples.md`). A LoRA card rarely embeds images in
  * its text: it lists them in its YAML header's `widget` — an image or video
  * and the prompt that made it — and HuggingFace renders them where the README
  * says `<Gallery />`. Many other repositories keep sample images as files
  * (`images/`, `samples/`, `assets/`). Both come in a model listing expanded
  * with `cardData` and `siblings`, and in a model's own info.
  */
object HuggingFaceExamples {
  private[routes] case class WidgetOutput(url: Option[String] = None)
  private[routes] case class Widget(
      text: Option[String] = None,
      output: Option[WidgetOutput] = None
  )
  private[routes] case class CardData(widget: List[Widget] = Nil)
  private[routes] case class Sibling(rfilename: String)

  /** The parts of a listing entry, or of a model's info, examples come from. */
  private[routes] case class Sources(
      id: String,
      cardData: Option[CardData] = None,
      siblings: List[Sibling] = Nil
  )

  private val widgetCodec: JsonValueCodec[Widget] = JsonCodecMaker.make

  /** `widget` is YAML written by hand: a list as HuggingFace documents it, at
    * times a single entry; an entry that is not an object is skipped.
    */
  private given widgetsCodec: JsonValueCodec[List[Widget]] =
    new JsonValueCodec[List[Widget]] {
      val nullValue: List[Widget] = Nil

      def encodeValue(x: List[Widget], out: JsonWriter): Unit = out.writeNull()

      def decodeValue(in: JsonReader, default: List[Widget]): List[Widget] =
        if (in.isNextToken('[')) {
          if (in.isNextToken(']')) Nil
          else {
            in.rollbackToken()
            val entries = List.newBuilder[Widget]
            while ({
              entries ++= entry(in)
              in.isNextToken(',')
            }) ()
            if (in.isCurrentToken(']')) entries.result()
            else in.arrayEndOrCommaError()
          }
        } else {
          in.rollbackToken()
          entry(in).toList
        }

      private def entry(in: JsonReader): Option[Widget] =
        if (in.isNextToken('{')) {
          in.rollbackToken()
          Some(widgetCodec.decodeValue(in, widgetCodec.nullValue))
        } else {
          in.rollbackToken()
          in.skip()
          None
        }
    }

  private given sourcesCodec: JsonValueCodec[Sources] = JsonCodecMaker.make
  private given listingCodec: JsonValueCodec[List[Sources]] =
    JsonCodecMaker.make

  /** A listing's examples by repository. A card too odd to read costs its page
    * its examples, never the listing itself, which is read apart.
    */
  def ofListing(json: String): Map[String, List[ModelExample]] =
    try
      readFromString[List[Sources]](json)
        .map(sources => sources.id -> of(sources))
        .toMap
    catch { case NonFatal(_) => Map.empty }

  /** One model info's examples, or none when its card cannot be read. */
  def ofModel(json: String): List[ModelExample] =
    sourcesOf(json).map(of).getOrElse(Nil)

  /** The card's gallery alone: what HuggingFace puts where `<Gallery />` is. */
  def galleryOf(json: String): List[ModelExample] =
    sourcesOf(json).map(card).getOrElse(Nil)

  private def sourcesOf(json: String): Option[Sources] =
    try Some(readFromString[Sources](json))
    catch { case NonFatal(_) => None }

  /** The card's gallery first, in its order and with its prompts, then the
    * repository's other media files.
    */
  private def of(sources: Sources): List[ModelExample] = {
    val gallery = card(sources)
    val shown = gallery.map(example => decoded(example.url)).toSet
    val files = sources.siblings
      .map(_.rfilename)
      .filter(isMedia)
      .map(name => ModelExample(fileUrl(sources.id, name), isVideo(name)))
      .filterNot(example => shown.contains(decoded(example.url)))
    (gallery ++ files).distinctBy(_.url)
  }

  private def card(sources: Sources): List[ModelExample] =
    sources.cardData.toList.flatMap(_.widget).flatMap { widget =>
      widget.output
        .flatMap(_.url)
        .map(_.trim)
        .filter(_.nonEmpty)
        .map(url =>
          ModelExample(
            url = absolute(sources.id, url),
            video = isVideo(url.takeWhile(_ != '?')),
            // Cards without prompts write "-".
            prompt = widget.text.map(_.trim).filter(t => t.nonEmpty && t != "-")
          )
        )
    }

  private val ImageExtensions = List(".png", ".jpg", ".jpeg", ".webp", ".gif")
  private val VideoExtensions = List(".mp4", ".webm", ".mov")

  private def isVideo(path: String): Boolean =
    VideoExtensions.exists(path.toLowerCase.endsWith)

  private def isMedia(path: String): Boolean =
    isVideo(path) || ImageExtensions.exists(path.toLowerCase.endsWith)

  /** A file of the repository, its path encoded segment by segment. */
  private def fileUrl(repo: String, path: String): String =
    URI(
      "https",
      "huggingface.co",
      s"/$repo/resolve/main/$path",
      null
    ).toASCIIString

  /** A widget URL may be relative to the repository, and encoded or not. */
  private def absolute(repo: String, url: String): String =
    if (url.startsWith("https://") || url.startsWith("http://")) url
    else fileUrl(repo, decoded(url.stripPrefix("./").stripPrefix("/")))

  private def decoded(url: String): String =
    try URLDecoder.decode(url, StandardCharsets.UTF_8)
    catch { case NonFatal(_) => url }
}
