package drift.frontend.components

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.StringAsIsCodec

/** HTML the backend rendered and cleaned (`specs/24`: HuggingFace model cards,
  * Civitai descriptions), inserted as-is. The one place drift sets `innerHTML`:
  * never hand it anything that did not come through the backend's
  * `ModelDescriptions`.
  */
class RichText(html: String) extends Component {
  private val innerHtml = htmlProp[String, String]("innerHTML", StringAsIsCodec)

  lazy val element: HtmlElement =
    div(cls := "content rich-text", innerHtml := html)
}
