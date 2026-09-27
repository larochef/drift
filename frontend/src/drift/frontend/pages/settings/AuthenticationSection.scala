package drift.frontend.pages.settings

import drift.frontend.components.*
import drift.frontend.services.AuthTokenService
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** The API tokens drift downloads with: the saved ones, which is active per
  * provider, and a modal to add one — like adding a runtime.
  */
class AuthenticationSection(authTokenService: AuthTokenService)
    extends Component {

  /** The add modal's fields. They outlive the modal, so closing and reopening
    * does not lose what was typed; saving clears them and closes it.
    */
  private val showAddToken = Var(false)
  private val newTokenProvider = Var[AuthProvider](AuthProvider.Civitai)
  private val newTokenLabel = Var("")
  private val newTokenValue = Var("")

  private def providerLabel(provider: AuthProvider): String = provider match {
    case AuthProvider.Civitai     => "Civitai"
    case AuthProvider.HuggingFace => "HuggingFace"
    case AuthProvider.ModelScope  => "ModelScope"
  }

  /** Enough to recognize a token, not enough to shoulder-surf it. */
  private def masked(token: String): String =
    if (token.length <= 8) "••••"
    else s"${token.take(4)}…${token.takeRight(4)}"

  private def activeId(
      selection: AuthTokenSelection,
      provider: AuthProvider
  ): Option[String] = provider match {
    case AuthProvider.Civitai     => selection.civitaiTokenId
    case AuthProvider.HuggingFace => selection.huggingFaceTokenId
    case AuthProvider.ModelScope  => selection.modelScopeTokenId
  }

  private def tokenRow(
      token: AuthToken,
      selection: AuthTokenSelection
  ): HtmlElement = {
    val isActive = activeId(selection, token.provider).contains(token.id)
    div(
      cls := "level is-mobile mb-1 is-marginless",
      div(
        cls := "level-left",
        div(
          p(
            cls := "text-primary",
            strong(token.label),
            span(
              cls := "tag is-dark is-small ml-2",
              providerLabel(token.provider)
            ),
            span(cls := "text-secondary is-size-7 ml-2", masked(token.token)),
            if (isActive)
              span(cls := "tag is-info is-small ml-2", "active")
            else emptyNode
          )
        )
      ),
      div(
        cls := "level-right",
        if (!isActive)
          button(
            cls := "button is-small mr-1",
            "Make active",
            onClick --> { _ =>
              authTokenService.push(
                AuthTokenService.Command
                  .SetActive(token.provider, Some(token.id))
              )
            }
          )
        else emptyNode,
        button(
          cls := "button is-danger is-small",
          "🗑️",
          title := "Delete token",
          onClick --> { _ =>
            if (
              window.confirm(
                s"Delete token '${token.label}' (${providerLabel(token.provider)})?"
              )
            ) authTokenService.push(AuthTokenService.Command.Delete(token.id))
          }
        )
      )
    )
  }

  /** Saves the typed token under an id made from its label — unique among the
    * saved ones — then clears the fields and closes the modal.
    */
  private def save(tokens: List[AuthToken]): Unit = {
    val label = newTokenLabel.now().trim
    val token = newTokenValue.now().trim
    if (label.nonEmpty && token.nonEmpty) {
      val slug = label.toLowerCase
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
            label = label,
            provider = newTokenProvider.now(),
            token = token,
            createdAt = System.currentTimeMillis()
          )
        )
      )
      newTokenLabel.set("")
      newTokenValue.set("")
      showAddToken.set(false)
    }
  }

  private def addTokenModal: Modifier[HtmlElement] =
    child <-- showAddToken.signal.map {
      case false => emptyNode
      case true  =>
        div(
          cls := "modal is-active",
          ScrollLock.whileMounted,
          documentEvents(_.onKeyDown).filter(_.key == "Escape")
            --> (_ => showAddToken.set(false)),
          div(
            cls := "modal-background",
            onClick --> (_ => showAddToken.set(false))
          ),
          div(
            cls := "modal-card",
            headerTag(
              cls := "modal-card-head",
              p(cls := "modal-card-title", "Add a token"),
              button(
                cls := "delete",
                aria.label := "close",
                onClick --> (_ => showAddToken.set(false))
              )
            ),
            sectionTag(
              cls := "modal-card-body",
              div(
                cls := "field",
                label(cls := "label text-primary", "Provider"),
                div(
                  cls := "select",
                  select(
                    onChange.mapToValue --> Observer[String] { name =>
                      AuthProvider.values
                        .find(_.toString == name)
                        .foreach(newTokenProvider.set)
                    },
                    AuthProvider.values.toList.map(provider =>
                      option(
                        value := provider.toString,
                        if (provider == newTokenProvider.now()) selected := true
                        else emptyNode,
                        providerLabel(provider)
                      )
                    )
                  )
                )
              ),
              div(
                cls := "field",
                label(cls := "label text-primary", "Label"),
                input(
                  cls := "input",
                  placeholder := "e.g. main account",
                  controlled(
                    value <-- newTokenLabel.signal,
                    onInput.mapToValue --> newTokenLabel
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
                    value <-- newTokenValue.signal,
                    onInput.mapToValue --> newTokenValue
                  )
                )
              )
            ),
            footerTag(
              cls := "modal-card-foot",
              child <-- authTokenService.tokens.map { tokens =>
                button(
                  cls := "button is-primary",
                  "Save token",
                  disabled <-- newTokenLabel.signal
                    .combineWith(newTokenValue.signal)
                    .map { (label, token) =>
                      label.trim.isEmpty || token.trim.isEmpty
                    },
                  onClick --> (_ => save(tokens))
                )
              },
              button(
                cls := "button",
                "Cancel",
                onClick --> (_ => showAddToken.set(false))
              )
            )
          )
        )
    }

  lazy val element: HtmlElement = div(
    div(
      cls := "level mt-5",
      div(
        cls := "level-left",
        h2(cls := "is-size-5 text-primary", "Authentication")
      ),
      div(
        cls := "level-right",
        button(
          cls := "button is-primary is-small",
          span(cls := "plus-icon", "+"),
          " Add a token",
          onClick --> (_ => showAddToken.set(true))
        )
      )
    ),
    p(
      cls := "text-secondary is-size-7",
      "API tokens for gated downloads (Civitai answers 401 without one) and " +
        "private or gated repositories on HuggingFace and ModelScope. Stored " +
        "as plain text under ~/.config/drift/auth-tokens/. The active token " +
        "applies immediately — no restart; with none active, the " +
        "CIVITAI_API_TOKEN / HF_TOKEN / MODELSCOPE_API_TOKEN environment " +
        "variables are the fallback."
    ),
    children <-- authTokenService.tokens
      .combineWith(authTokenService.selection)
      .map { (tokens, selection) =>
        if (tokens.isEmpty)
          List(
            p(
              cls := "text-secondary",
              "No tokens saved yet. \"Add a token\" stores one."
            )
          )
        else tokens.map(tokenRow(_, selection))
      },
    addTokenModal
  )
}
