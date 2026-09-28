package drift.frontend.components

import drift.frontend.services.AuthTokenService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The fields of a new API token and the button saving it: the Settings modal's
  * body, and what the Civitai browser shows in place of its search while no
  * token is set. With `provider` fixed, the provider select is left out.
  *
  * The fields live as long as the form, so a modal closed and reopened keeps
  * what was typed; saving clears them.
  */
class AuthTokenForm(
    authTokenService: AuthTokenService,
    provider: Option[AuthProvider] = None,
    initialLabel: String = ""
) {
  private val chosenProvider =
    Var[AuthProvider](provider.getOrElse(AuthProvider.Civitai))
  private val tokenLabel = Var(initialLabel)
  private val token = Var("")

  /** Saves the typed token under an id made from its label — unique among the
    * saved ones — then clears the fields. The service makes the first token of
    * a provider its active one.
    */
  private def save(tokens: List[AuthToken], onSaved: () => Unit): Unit = {
    val typedLabel = tokenLabel.now().trim
    val typedToken = token.now().trim
    if (typedLabel.nonEmpty && typedToken.nonEmpty) {
      val slug = typedLabel.toLowerCase
        .map(c => if (c.isLetterOrDigit) c else '-')
        .split('-')
        .filter(_.nonEmpty)
        .mkString("-") match {
        case ""    => "token"
        case other => other
      }
      val taken = tokens.map(_.id).toSet
      val id =
        if (!taken(slug)) slug
        else
          Iterator
            .from(2)
            .map(n => s"$slug-$n")
            .filterNot(taken)
            .next()
      authTokenService.push(
        AuthTokenService.Command.Create(
          AuthToken(
            id = id,
            label = typedLabel,
            provider = chosenProvider.now(),
            token = typedToken,
            createdAt = System.currentTimeMillis()
          )
        )
      )
      tokenLabel.set(initialLabel)
      token.set("")
      onSaved()
    }
  }

  lazy val fields: HtmlElement = div(
    provider match {
      case Some(_) => emptyNode
      case None    =>
        div(
          cls := "field",
          label(cls := "label text-primary", "Provider"),
          div(
            cls := "select",
            select(
              onChange.mapToValue --> Observer[String] { name =>
                AuthProvider.values
                  .find(_.toString == name)
                  .foreach(chosenProvider.set)
              },
              AuthProvider.values.toList.map(one =>
                option(
                  value := one.toString,
                  if (one == chosenProvider.now()) selected := true
                  else emptyNode,
                  AuthTokenForm.providerLabel(one)
                )
              )
            )
          )
        )
    },
    div(
      cls := "field",
      label(cls := "label text-primary", "Label"),
      input(
        cls := "input",
        placeholder := "e.g. main account",
        controlled(
          value <-- tokenLabel.signal,
          onInput.mapToValue --> tokenLabel
        )
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary", "Token"),
      input(
        cls := "input",
        placeholder := "token",
        controlled(
          value <-- token.signal,
          onInput.mapToValue --> token
        )
      )
    )
  )

  def saveButton(onSaved: () => Unit): Modifier[HtmlElement] =
    child <-- authTokenService.tokens.map { tokens =>
      button(
        cls := "button is-primary",
        "Save token",
        disabled <-- tokenLabel.signal
          .combineWith(token.signal)
          .map((typedLabel, typedToken) =>
            typedLabel.trim.isEmpty || typedToken.trim.isEmpty
          ),
        onClick --> (_ => save(tokens, onSaved))
      )
    }
}

object AuthTokenForm {
  def providerLabel(provider: AuthProvider): String = provider match {
    case AuthProvider.Civitai     => "Civitai"
    case AuthProvider.HuggingFace => "HuggingFace"
    case AuthProvider.ModelScope  => "ModelScope"
  }
}
