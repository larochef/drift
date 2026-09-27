package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object AuthTokenService {
  enum Command {
    case Load
    case Create(token: AuthToken)
    case Delete(id: String)

    /** `None` deactivates the provider's slot, falling back to the environment
      * variable.
      */
    case SetActive(provider: AuthProvider, tokenId: Option[String])
  }
}

class AuthTokenService extends ServiceErrors {
  import AuthTokenService.Command

  private val listFn = ApiClient.stream(drift.shared.listAuthTokens)
  private val createFn = ApiClient.stream(drift.shared.createAuthToken)
  private val deleteFn = ApiClient.stream(drift.shared.deleteAuthToken)
  private val getSelectionFn =
    ApiClient.stream(drift.shared.getAuthTokenSelection)
  private val setSelectionFn =
    ApiClient.stream(drift.shared.setAuthTokenSelection)

  private val _tokens = Var(List.empty[AuthToken])
  private val _selection = Var(AuthTokenSelection())

  val tokens: Signal[List[AuthToken]] = _tokens.signal
  val selection: Signal[AuthTokenSelection] = _selection.signal

  private val cmdBus = new EventBus[Command]

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  private def activeId(
      selection: AuthTokenSelection,
      provider: AuthProvider
  ): Option[String] = provider match {
    case AuthProvider.Civitai     => selection.civitaiTokenId
    case AuthProvider.HuggingFace => selection.huggingFaceTokenId
    case AuthProvider.ModelScope  => selection.modelScopeTokenId
  }

  // See ModelService.effects for why writes merge instead of switching.
  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ =>
        listFn(()).combineWith(getSelectionFn(())).recoverToTry
      )
      --> Observer[Try[(List[AuthToken], AuthTokenSelection)]] {
        case Success((tokens, selection)) =>
          clearError()
          _tokens.set(tokens.sortBy(t => (t.provider.toString, t.label)))
          _selection.set(selection)
        case Failure(err) => reportFailure("Loading tokens", err)
      },
    cmdBus.events
      .collect { case Command.Create(token) => token }
      .flatMapMerge(token => createFn(token).recoverToTry)
      --> Observer[Try[AuthToken]] {
        case Success(token) =>
          clearError()
          _tokens.update(all =>
            (all.filterNot(_.id == token.id) :+ token)
              .sortBy(t => (t.provider.toString, t.label))
          )
          // The first token for a provider becomes its active one; switching
          // afterwards is explicit.
          if (activeId(_selection.now(), token.provider).isEmpty)
            push(Command.SetActive(token.provider, Some(token.id)))
        case Failure(err) => reportFailure("Saving the token", err)
      },
    cmdBus.events
      .collect { case Command.Delete(id) => id }
      .flatMapMerge(id =>
        deleteFn(id).map(deleted => (id, deleted)).recoverToTry
      )
      --> Observer[Try[(String, Boolean)]] {
        case Success((id, true)) =>
          clearError()
          _tokens.update(_.filterNot(_.id == id))
          // Mirrors the backend's cleanup: a deleted token is no longer
          // active anywhere.
          _selection.update(current =>
            current.copy(
              civitaiTokenId = current.civitaiTokenId.filterNot(_ == id),
              huggingFaceTokenId =
                current.huggingFaceTokenId.filterNot(_ == id),
              modelScopeTokenId = current.modelScopeTokenId.filterNot(_ == id)
            )
          )
        case Success((id, false)) =>
          reportFailure(
            "Deleting the token",
            s"the server refused to delete '$id'."
          )
        case Failure(err) => reportFailure("Deleting the token", err)
      },
    cmdBus.events
      .collect { case Command.SetActive(provider, id) => (provider, id) }
      .flatMapMerge { (provider, id) =>
        val next = provider match {
          case AuthProvider.Civitai =>
            _selection.now().copy(civitaiTokenId = id)
          case AuthProvider.HuggingFace =>
            _selection.now().copy(huggingFaceTokenId = id)
          case AuthProvider.ModelScope =>
            _selection.now().copy(modelScopeTokenId = id)
        }
        setSelectionFn(next).recoverToTry
      }
      --> Observer[Try[AuthTokenSelection]] {
        case Success(selection) =>
          clearError()
          _selection.set(selection)
        case Failure(err) => reportFailure("Setting the active token", err)
      }
  )
}
