package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Which service a stored token authenticates against. */
enum AuthProvider derives CanEqual {
  case Civitai, HuggingFace, ModelScope
}
object AuthProvider {
  // String-encoded like the other singleton enums; the explicit schema keeps
  // tapir's derivation in agreement (see RuntimeBackend).
  given Schema[AuthProvider] =
    Schema.derivedEnumeration[AuthProvider].defaultStringBased
  given JsonValueCodec[List[AuthProvider]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** One saved API token. Several can coexist per provider (different accounts, a
  * throwaway key…); [[AuthTokenSelection]] names the active one.
  *
  * Stored as plain text under `~/.config/drift/auth-tokens/` — a deliberate
  * first-version trade: the directory is the user's own machine, and secrecy at
  * rest can come later without changing this shape.
  */
case class AuthToken(
    id: String,
    label: String,
    provider: AuthProvider,
    token: String,
    createdAt: Long
)
object AuthToken {
  // No discriminator: `AuthProvider` has only singleton cases, so it encodes
  // as a plain string ("Civitai"). The config must be inline -- the macro
  // needs a constant expression.
  given JsonValueCodec[AuthToken] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[AuthToken]] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[AuthToken]] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
}

/** The active token per provider. An absent side means "fall back to the
  * environment variable" (`CIVITAI_API_TOKEN` / `HF_TOKEN` /
  * `MODELSCOPE_API_TOKEN`), which stays the bootstrap path.
  */
case class AuthTokenSelection(
    civitaiTokenId: Option[String],
    huggingFaceTokenId: Option[String],
    modelScopeTokenId: Option[String]
)
object AuthTokenSelection {

  /** No token chosen for any site. */
  val none: AuthTokenSelection = AuthTokenSelection(
    civitaiTokenId = None,
    huggingFaceTokenId = None,
    modelScopeTokenId = None
  )

  given JsonValueCodec[AuthTokenSelection] = JsonCodecMaker.make(storedRecord)
}

val listAuthTokens: PublicEndpoint[Unit, Unit, List[AuthToken], Any] =
  base.get.in("auth-tokens").out(jsonBody[List[AuthToken]])

val getAuthToken: PublicEndpoint[String, Unit, Option[AuthToken], Any] =
  base.get
    .in("auth-tokens" / path[String])
    .out(jsonBody[Option[AuthToken]])

val createAuthToken: PublicEndpoint[AuthToken, Unit, AuthToken, Any] =
  base.post
    .in("auth-tokens")
    .in(jsonBody[AuthToken])
    .out(jsonBody[AuthToken])

val updateAuthToken
    : PublicEndpoint[(String, AuthToken), Unit, Option[AuthToken], Any] =
  base.put
    .in("auth-tokens" / path[String])
    .in(jsonBody[AuthToken])
    .out(jsonBody[Option[AuthToken]])

val deleteAuthToken: PublicEndpoint[String, Unit, Boolean, Any] =
  base.delete.in("auth-tokens" / path[String]).out(jsonBody[Boolean])

val getAuthTokenSelection: PublicEndpoint[Unit, Unit, AuthTokenSelection, Any] =
  base.get.in("auth-token-selection").out(jsonBody[AuthTokenSelection])

/** The providers whose environment variable carries a token — the fallback a
  * browser counts as a token, since the frontend cannot read the backend's
  * environment. Never the values themselves.
  */
val getEnvironmentAuthProviders
    : PublicEndpoint[Unit, Unit, List[AuthProvider], Any] =
  base.get
    .in("auth-token-environment")
    .out(jsonBody[List[AuthProvider]])

/** Takes effect immediately: the download paths resolve the active token per
  * request rather than at startup, so no restart is needed.
  */
val setAuthTokenSelection
    : PublicEndpoint[AuthTokenSelection, Unit, AuthTokenSelection, Any] =
  base.put
    .in("auth-token-selection")
    .in(jsonBody[AuthTokenSelection])
    .out(jsonBody[AuthTokenSelection])
