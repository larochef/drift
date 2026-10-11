package drift.protocol

import drift.protocol.Requests.given

import scala.util.*

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.model.StatusCode
import sttp.tapir.*
import sttp.tapir.json.jsoniter.*

/** Why a server did not do what it was asked: `{"error": "…"}`. */
case class JobError(error: String)
object JobError {
  given JsonValueCodec[JobError] = JsonCodecMaker.make
}

/** The job protocol as endpoints: what the drift runner serves, and what drives
  * a server builds its calls from.
  */
object JobEndpoints {

  /** A request as it arrived: read, or why it could not be — malformed, or
    * naming a field the protocol lacks. The reason is the server's to answer
    * with, in the protocol's own words, so reading never fails the call.
    */
  type Asked[T] = Either[String, T]

  // Pictures and videos travel as base64 text inside a request: far above the
  // 4 MiB a JSON string is read up to unless told otherwise.
  private val readerConfig = ReaderConfig.withMaxCharBufSize(256 * 1024 * 1024)

  private def asked[T: JsonValueCodec]: EndpointIO.Body[String, Asked[T]] =
    customCodecJsonBody[Asked[T]](using
      Codec.json[Asked[T]] { text =>
        DecodeResult.Value(Try(readFromString[T](text, readerConfig)) match {
          case Success(request) => Right(request)
          // jsoniter ends its message with where in the bytes, and a dump
          case Failure(error) =>
            Left(String.valueOf(error.getMessage).split(", offset: ").head)
        })
      } {
        case Right(request) => writeToString(request)
        case Left(_)        => "{}"
      }(using Schema.string[Asked[T]])
    )

  // What the bodies hold is said by their types and their codecs; the schemas
  // tapir asks for would only describe them a second time, for documentation
  // nothing generates.
  private given Schema[NativeJob] = Schema.string
  private given Schema[JobError] = Schema.string

  private val root: Endpoint[Unit, Unit, (StatusCode, JobError), Unit, Any] =
    endpoint
      .in("sdcpp" / "v1")
      .errorOut(statusCode.and(jsonBody[JobError]))

  private val submitted =
    statusCode(StatusCode.Accepted).and(jsonBody[NativeJob])

  /** What the server is and takes, as sd-server describes itself. */
  val capabilities = root.get.in("capabilities").out(stringJsonBody)

  val submitImage =
    root.post.in("img_gen").in(asked[ImageJobRequest]).out(submitted)

  val submitVideo =
    root.post.in("vid_gen").in(asked[VideoJobRequest]).out(submitted)

  val submitUpscale =
    root.post.in("upscale").in(asked[UpscaleJobRequest]).out(submitted)

  val job = root.get.in("jobs" / path[String]("id")).out(jsonBody[NativeJob])

  val cancel =
    root.post
      .in("jobs" / path[String]("id") / "cancel")
      .out(jsonBody[NativeJob])
}
