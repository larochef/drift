package drift.frontend.components

import com.raquo.laminar.api.L.*

class Showcase {

  lazy val element: HtmlElement = div(
    cls := "showcase",
    div(
      cls := "showcase-grid",
      Showcase.pictures.map(picture =>
        figure(
          cls := "showcase-tile",
          title := picture.prompt,
          img(
            src := s"/assets/showcase/${picture.file}",
            alt := picture.prompt,
            loadingAttr := "lazy",
            onError --> (event =>
              event.target match {
                case image: org.scalajs.dom.html.Image =>
                  image.parentElement.style.display = "none"
                case _ => ()
              }
            )
          ),
          figCaption(span(cls := "showcase-style", picture.style))
        )
      )
    ),
    p(
      cls := "text-secondary is-size-7 mt-2",
      "Made with drift on Krea 2, one of the starter models — hover one " +
        "for its prompt."
    )
  )
}

object Showcase {

  final case class Picture(
      file: String,
      style: String,
      prompt: String
  )

  val pictures: List[Picture] = List(
    Picture(
      "film-portrait.webp",
      "Film photo",
      "35mm film portrait of an old fisherman mending a net on a harbour " +
        "wall at golden hour, weathered face, warm light, shallow depth of " +
        "field, Kodak Portra grain"
    ),
    Picture(
      "watercolour-village.webp",
      "Watercolour",
      "loose watercolour painting of a hill village in Tuscany at dawn, " +
        "cypress trees, soft washes bleeding into the paper, white of the " +
        "paper left for the sky"
    ),
    Picture(
      "anime-rooftop.webp",
      "Anime",
      "anime key visual, a girl in a yellow raincoat on a rooftop at night " +
        "looking over a neon city in the rain, reflections, cinematic " +
        "composition"
    ),
    Picture(
      "mars-poster.webp",
      "Poster & lettering",
      "retro 1950s travel poster, a rocket landing among red canyons, the " +
        "title \"VISIT MARS\" in bold art deco letters, flat colours, " +
        "screen-print texture"
    ),
    Picture(
      "clay-bookshop.webp",
      "3D clay",
      "isometric 3D render of a tiny cosy bookshop in soft clay style, warm " +
        "lamps, a cat asleep on a stack of books, pastel colours, studio " +
        "lighting"
    ),
    Picture(
      "oil-dragon.webp",
      "Oil painting",
      "romantic oil painting of a dragon gliding over a misty mountain lake " +
        "at sunrise, in the manner of the Hudson River School, visible " +
        "brushstrokes"
    ),
    Picture(
      "pencil-owl.webp",
      "Pencil drawing",
      "detailed graphite pencil drawing of a barn owl perched on a fence " +
        "post, fine cross-hatching, sketchbook paper texture, study " +
        "annotations in the margin"
    ),
    Picture(
      "pixel-cabin.webp",
      "Pixel art",
      "pixel art of a log cabin in a snowy forest at night, warm windows, " +
        "smoke from the chimney, aurora in the sky, 16-bit palette"
    )
  )
}
