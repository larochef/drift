package drift.frontend.components

import drift.shared.ModelExample

import com.raquo.laminar.api.L.*

/** One result tile — the same one in the three browsers (`specs/24`, `37`).
  *
  * It takes what a result *is*, never how to draw it: the previews fill the
  * tile through `BrowserMedia`, how many there are and what drift already
  * installed from it are the chips in the top-left corner, the media's own
  * badges hold the top-right, and the name, whoever published it and its
  * numbers sit over a gradient at the bottom. The sites count different things
  * — Civitai's thumbs up, HuggingFace's likes, ModelScope's stars — so the word
  * for it is a parameter, while the tile reads the same everywhere.
  */
class BrowserCard(
    /** The previews: the first fills the tile, the others stand in when it does
      * not load.
      */
    examples: List[ModelExample],
    /** How many the result has, which is not always how many came with it: the
      * repository sites count them without shipping them all.
      */
    exampleCount: Int,
    name: String,
    onOpen: () => Unit,
    /** The LoRAs this architecture already has, and where to look this result
      * up among them.
      */
    installed: Signal[Installed] = Val(Installed.none),
    installedAs: Installed => List[InstalledItem] = _ => Nil,
    /** Under the name: who published it. */
    author: Option[String] = None,
    downloads: Option[Long] = None,
    /** What the site counts as approval, and what it calls it. */
    likes: Option[Long] = None,
    likesLabel: String = "likes",
    tooltip: Option[String] = None
) extends Component {

  lazy val element: HtmlElement = div(
    cls := "civitai-card cursor-pointer",
    tooltip.map(text => title := text),
    onClick --> (_ => onOpen()),
    if (examples.isEmpty) BrowserCard.noPreview
    else BrowserMedia(examples, description = name).element,
    div(
      cls := "civitai-card-chips",
      BrowserCard.examplesChip(exampleCount),
      Installed.mark(installed, installedAs)
    ),
    div(
      cls := "civitai-card-overlay",
      p(cls := "civitai-card-title", name),
      author.map(who => p(cls := "civitai-card-creator", s"by $who")),
      div(
        cls := "civitai-card-stats",
        stat("⬇", downloads, "downloads"),
        stat("♥", likes, likesLabel)
      )
    )
  )

  /** A count worth showing: a result with none of something says nothing about
    * it rather than showing a zero.
    */
  private def stat(
      icon: String,
      count: Option[Long],
      what: String
  ): Option[HtmlElement] =
    count
      .filter(_ > 0)
      .map(value =>
        span(
          title := s"${value.toString} $what",
          s"$icon ${BrowserUtils.formatCount(value)}"
        )
      )
}

object BrowserCard {

  /** The tile of a result with no preview at all. */
  def noPreview: HtmlElement =
    div(cls := "civitai-card-media-missing is-size-7", "no example")

  /** A repository id as a tile shows it: its name over the gradient and its
    * owner as the byline, the way Civitai shows a model and its creator. The
    * whole `owner/name` stays in the tile's tooltip.
    */
  def ownerAndName(id: String): (Option[String], String) =
    id.split("/", 2) match {
      case Array(owner, name) => (Some(owner), name)
      case _                  => (None, id)
    }

  /** The corner chip: what there is to look at before opening the result. */
  private def examplesChip(count: Int): Option[HtmlElement] =
    Option.when(count > 0)(
      span(
        cls := "tag is-small is-dark",
        if (count == 1) "1 example" else s"$count examples"
      )
    )
}
