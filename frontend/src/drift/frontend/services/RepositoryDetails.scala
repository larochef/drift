package drift.frontend.services

import com.raquo.laminar.api.L.*

/** The opened repository itself (`specs/24`, `specs/37-modelscope.md`), as the
  * two repository services hold it: its files and examples, whether they are
  * loading, why they did not load, and which repository was asked for — a Retry
  * asks for exactly that one again.
  *
  * ModelScope answers a tree request with its own failure often enough that a
  * Retry is what the modal needs ("获取模型目录树失败", François, 2026-09-22): the next
  * attempt usually works, and until it does the repository shows no file and no
  * picture, which reads like an empty repository rather than a failure.
  */
trait RepositoryDetails[Detail] {

  protected val _detail = Var(Option.empty[Detail])

  /** `None` while it loads, and when the load failed. */
  val detail: Signal[Option[Detail]] = _detail.signal

  protected val _loadingDetail = Var(false)
  val loadingDetail: Signal[Boolean] = _loadingDetail.signal

  /** Why the repository did not load, shown in the modal beside a Retry. */
  protected val _detailFailure = Var(Option.empty[String])
  val detailFailure: Signal[Option[String]] = _detailFailure.signal

  private val detailRequest = Var(Option.empty[String])

  def retryDetail(): Unit = detailRequest.now().foreach(loadDetail)

  /** Sends the service's own `LoadDetail` command. */
  protected def loadDetail(repo: String): Unit

  protected def detailRequested(repo: String): Unit = {
    detailRequest.set(Some(repo))
    _detail.set(None)
    _detailFailure.set(None)
    _loadingDetail.set(true)
  }

  protected def detailArrived(found: Option[Detail]): Unit = {
    _detail.set(found)
    _loadingDetail.set(false)
  }

  protected def detailFailed(reason: String): Unit = {
    _loadingDetail.set(false)
    _detailFailure.set(Some(reason))
  }
}
