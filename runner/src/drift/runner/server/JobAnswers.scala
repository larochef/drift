package drift.runner.server

import drift.protocol.{JobError, NativeJob}

import sttp.model.StatusCode

/** What the native API answers about a job with. */
private[server] object JobAnswers {

  /** An answer to a job's submission, poll or cancel: its document, or the
    * status and reason it is refused with.
    */
  type Answer = Either[(StatusCode, JobError), NativeJob]

  def failed(status: Int, reason: String): Answer =
    Left(StatusCode(status) -> JobError(reason))
}
