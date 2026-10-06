package drift.shared

import scala.util.*

import com.github.plokhotnyuk.jsoniter_scala.core.*
import sttp.tapir.*
import sttp.tapir.DecodeResult.Error.JsonDecodeException

/** A JSON body that carries pictures or videos as base64 text — an import, a
  * generation's input images. `jsonBody` reads with jsoniter's defaults, which
  * refuse a string above 4 MiB: an 8 MB PNG was "too long string exceeded
  * 'maxCharBufSize'" (`bugs/41`).
  */
object LargeJsonBody {
  private val readerConfig = ReaderConfig.withMaxCharBufSize(256 * 1024 * 1024)

  def apply[T: JsonValueCodec: Schema]: EndpointIO.Body[String, T] =
    customCodecJsonBody[T](using
      Codec.json[T] { text =>
        Try(readFromString[T](text, readerConfig)) match {
          case Success(value) => DecodeResult.Value(value)
          case Failure(error) =>
            DecodeResult.Error(text, JsonDecodeException(List.empty, error))
        }
      }(writeToString[T](_))
    )
}
