package drift.frontend.services

import scala.concurrent.{ExecutionContext, Future}
import scala.concurrent.duration.{DurationInt, FiniteDuration}
import scala.scalajs.concurrent.JSExecutionContext
import scala.scalajs.js.JSConverters.*

import com.raquo.laminar.api.L.*
import sttp.client3.FetchBackend
import sttp.model.Uri
import sttp.tapir.PublicEndpoint
import sttp.tapir.client.sttp.SttpClientInterpreter

/** The single place the tapir/sttp wiring lives.
  *
  * Every service used to carry its own `FetchBackend`, base uri, interpreter
  * and `Future`-to-`EventStream` bridge. Eight copies meant eight places to get
  * error handling right, which is how the search paths ended up without the
  * `recoverToTry` the entity services had (see
  * `bugs/14-silent-failure-and-dead-pipeline.md`).
  *
  * `toClientThrowErrors` rejects the promise on any non-2xx, so every stream
  * this hands out can fail; callers are expected to `recoverToTry` and report
  * through [[ServiceErrors]] rather than let the stream end.
  */
object ApiClient {
  private given ExecutionContext = JSExecutionContext.Implicits.queue

  private val backend = FetchBackend()
  private val baseUri = Some(Uri.unsafeParse("/"))
  private val interpreter = SttpClientInterpreter()

  /** sttp's own wait for an answer. */
  private val DefaultWait: FiniteDuration = 1.minute

  /** The raw client for an endpoint, for callers that need the `Future`. */
  def call[I, O](endpoint: PublicEndpoint[I, Unit, O, Any]): I => Future[O] =
    interpreter.toClientThrowErrors(endpoint, baseUri, backend)

  /** The client for an endpoint as a single-value `EventStream`. */
  def stream[I, O](
      endpoint: PublicEndpoint[I, Unit, O, Any]
  ): I => EventStream[O] = {
    val fn = call(endpoint)
    input => EventStream.fromJsPromise(fn(input).toJSPromise)
  }

  /** `stream` for an endpoint whose failures carry a reason meant for the user
    * (`upstreamFailure`, `specs/24`): the stream fails with that reason alone,
    * instead of the interpreter's "Endpoint … returned error: …, inputs: …".
    */
  def streamWithFailureReason[I, O](
      endpoint: PublicEndpoint[I, String, O, Any],
      /** How long the answer may take. sttp gives a request one minute and then
        * aborts the fetch — "the operation was aborted" — which is far less
        * than an assistant takes to read a large picture.
        */
      within: FiniteDuration = DefaultWait
  ): I => EventStream[O] = {
    val request = interpreter.toRequestThrowDecodeFailures(endpoint, baseUri)
    input =>
      EventStream.fromJsPromise(
        request(input)
          .readTimeout(within)
          .send(backend)
          .map(_.body)
          .flatMap {
            case Right(output) => Future.successful(output)
            case Left(reason)  =>
              Future.failed(
                RuntimeException(
                  // An empty reason is drift's own server giving up before the
                  // site answered: it sends a 503 with no body.
                  if (reason.trim.nonEmpty) reason
                  else "No answer came in time — the site may be overloaded."
                )
              )
          }
          .toJSPromise
      )
  }
}
