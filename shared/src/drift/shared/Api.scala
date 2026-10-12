package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*

// jsoniter derives a codec per concrete type rather than composing them, so a
// companion declares exactly the shapes -- `T`, `List[T]`, `Option[T]` -- that
// the endpoints and `StorageService` round-trip. A type that only ever appears
// inside another is inlined into its parent's codec and needs none of its own.

// `delete*` answers with a bare JSON boolean, and primitives have no codec in
// jsoniter's implicit scope either.
given JsonValueCodec[Boolean] = JsonCodecMaker.make
given JsonValueCodec[List[String]] = JsonCodecMaker.make

/** How a stored record is written and read: every field, an empty list and an
  * absent value included, so a file says all of what it holds. A record lacking
  * a field does not decode — a field added to an entity comes with a migration
  * that adds it to the stored records. The one field a file may still omit is
  * an optional one, read as absent.
  */
inline def storedRecord: CodecMakerConfig = CodecMakerConfig
  .withTransientDefault(false)
  .withTransientEmpty(false)
  .withTransientNone(false)
  .withRequireCollectionFields(true)
  .withRequireDefaultFields(true)

/** Every endpoint's `/api` prefix, which every file of the package builds on:
  * Vite's dev proxy forwards nothing else, and the backend answers any other
  * path with the page's shell.
  */
private[shared] val base = endpoint.in("api")

// The entities' endpoints are `lazy` for one reason: a plain top-level `val`
// initializes in its file's static initializer, and tapir's auto-derived
// schemas are enough bytecode that all of them together overran the JVM's
// 64 KB limit on one method ("Method too large:
// drift/shared/Api$package$.<clinit>") the day `Model` and `RunConfiguration`
// each gained a field (`specs/16-parameter-resolution.md`). A lazy val gets
// its own initializer, so the ceiling is per endpoint rather than per file.

/** The error output of an endpoint relaying another site (`specs/24`): 502, and
  * the site's reason in words, which the page shows as it is — not the client's
  * "Endpoint … returned error: undefined, inputs: …".
  */
private[shared] val upstreamFailure: EndpointOutput[String] =
  statusCode(sttp.model.StatusCode.BadGateway).and(stringBody)
