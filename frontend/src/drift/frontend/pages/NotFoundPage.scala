package drift.frontend.pages

import drift.frontend.components.Component

import com.raquo.laminar.api.L.*

object NotFoundPage extends Component {

  // Fresh per render, same reason as HomePage.
  def element: HtmlElement = div(
    h1("404 - Not Found"),
    p("The page you're looking for doesn't exist.")
  )
}
