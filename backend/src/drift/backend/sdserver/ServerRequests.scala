package drift.backend.sdserver

import drift.shared.*

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.*

/** The bodies drift sends a server, as opposed to the ones it stores or answers
  * the page with.
  *
  * Every field is written, even one equal to its default: a server applies a
  * request on top of its launch flags, so a field left out means "what the
  * configuration launched with" — 20 steps asked of a server launched with
  * `--steps 4` ran 4, a 512 px tile came back at the server's own width
  * (`bugs/29`). The fields that are `None` are still left out: those are the
  * ones that mean "the server's own".
  */
private[backend] object ServerRequests {
  private inline def everyField = CodecMakerConfig
    .withDiscriminatorFieldName(None)
    .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
    .withTransientDefault(false)

  given JsonValueCodec[ImageGenerationParameters] =
    JsonCodecMaker.make(everyField)
  given JsonValueCodec[VideoGenerationParameters] =
    JsonCodecMaker.make(everyField)
}
