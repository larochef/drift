package drift.frontend.services

import com.raquo.laminar.api.L.*

/** The opened repository's card (`specs/24`), as the two repository services
  * hold it: the card, whether it is loading, why it did not load, and which
  * repository was asked for — a Retry asks for exactly that one again.
  */
trait RepositoryCards[Card] {

  protected val _card = Var(Option.empty[Card])

  /** `None` while it loads, and when the load failed. */
  val card: Signal[Option[Card]] = _card.signal

  protected val _loadingCard = Var(false)
  val loadingCard: Signal[Boolean] = _loadingCard.signal

  /** Why the card did not load, shown in its tab beside a Retry. */
  protected val _cardFailure = Var(Option.empty[String])
  val cardFailure: Signal[Option[String]] = _cardFailure.signal

  private val cardRequest = Var(Option.empty[String])

  def retryCard(): Unit = cardRequest.now().foreach(loadCard)

  /** Sends the service's own `LoadCard` command. */
  protected def loadCard(repo: String): Unit

  /** Before the request goes out: the tab says it is loading, and a Retry knows
    * what to ask for.
    */
  protected def cardRequested(repo: String): Unit = {
    cardRequest.set(Some(repo))
    _card.set(None)
    _cardFailure.set(None)
    _loadingCard.set(true)
  }

  protected def cardArrived(found: Card): Unit = {
    _card.set(Some(found))
    _loadingCard.set(false)
  }

  protected def cardFailed(reason: String): Unit = {
    _loadingCard.set(false)
    _cardFailure.set(Some(reason))
  }
}
