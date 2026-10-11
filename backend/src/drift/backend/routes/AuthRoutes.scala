package drift.backend.routes

import drift.backend.auth.AuthTokens
import drift.backend.storage.{Collections, StorageService}
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

def authEndpoints(
    storage: StorageService,
    tokens: AuthTokens
): List[ServerEndpoint[Any, Identity]] =
  endpointsFor[AuthToken](
    storage,
    Collections.authTokens,
    _.id,
    listAuthTokens,
    getAuthToken,
    createAuthToken,
    updateAuthToken,
    deleteAuthToken,
    onDeleted = tokens.cleanupDeleted
  ) ++ List(
    getAuthTokenSelection.serverLogicSuccess[Identity](_ => tokens.selection),
    setAuthTokenSelection.serverLogicSuccess[Identity](tokens.saveSelection),
    getEnvironmentAuthProviders.serverLogicSuccess[Identity](_ =>
      tokens.fromEnvironment
    )
  )
