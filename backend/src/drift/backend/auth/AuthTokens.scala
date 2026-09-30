package drift.backend.auth

import drift.backend.storage.StorageService
import drift.shared.*

/** Resolves the token the download paths should send, live per request — a
  * token saved in Settings works without a server restart, which is the whole
  * reason this is a lookup and not a startup value (a 401 mid-session used to
  * cost a restart).
  */
final class AuthTokens(storage: StorageService) {

  def selection: AuthTokenSelection =
    storage
      .get[AuthTokenSelection]("settings", "auth-token-selection")
      .getOrElse(AuthTokenSelection())

  def saveSelection(next: AuthTokenSelection): AuthTokenSelection =
    storage.save("settings", "auth-token-selection", next)

  /** Active stored token first; the environment variable stays the bootstrap
    * fallback for machines configured the old way.
    */
  def civitai: Option[String] =
    selection.civitaiTokenId
      .flatMap(tokenValue)
      .orElse(sys.env.get("CIVITAI_API_TOKEN"))

  def huggingFace: Option[String] =
    selection.huggingFaceTokenId
      .flatMap(tokenValue)
      .orElse(sys.env.get("HF_TOKEN"))

  /** `MODELSCOPE_API_TOKEN` is the variable ModelScope's own SDK reads. */
  def modelScope: Option[String] =
    selection.modelScopeTokenId
      .flatMap(tokenValue)
      .orElse(sys.env.get("MODELSCOPE_API_TOKEN"))

  /** The providers whose environment variable carries a non-blank token. */
  def fromEnvironment: List[AuthProvider] =
    List(
      AuthProvider.Civitai -> "CIVITAI_API_TOKEN",
      AuthProvider.HuggingFace -> "HF_TOKEN",
      AuthProvider.ModelScope -> "MODELSCOPE_API_TOKEN"
    ).collect {
      case (provider, variable)
          if sys.env.get(variable).exists(_.trim.nonEmpty) =>
        provider
    }

  private def tokenValue(id: String): Option[String] =
    storage
      .get[AuthToken]("auth-tokens", id)
      .map(_.token.trim)
      .filter(_.nonEmpty)

  /** A deleted token must not stay active — the selection drops any side
    * pointing at it, falling back to the environment variable.
    */
  def cleanupDeleted(token: AuthToken): Unit = {
    val current = selection
    val next = current.copy(
      civitaiTokenId = current.civitaiTokenId.filterNot(_ == token.id),
      huggingFaceTokenId = current.huggingFaceTokenId.filterNot(_ == token.id),
      modelScopeTokenId = current.modelScopeTokenId.filterNot(_ == token.id)
    )
    if (next != current) saveSelection(next)
  }
}
