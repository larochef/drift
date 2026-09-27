package drift.backend.routes

import scala.io.Source

import sttp.model.StatusCode
import sttp.shared.Identity
import sttp.tapir.*
import sttp.tapir.files.*
import sttp.tapir.server.ServerEndpoint

def frontendEndpoints: List[ServerEndpoint[Any, Identity]] = List(
  staticResourcesGetServerEndpoint[Identity]("assets")(
    Thread.currentThread().getContextClassLoader,
    "assets",
    options = FilesOptions.default[Identity].withUseGzippedIfAvailable
  ),
  endpoint.get
    .in(paths)
    .errorOut(statusCode(StatusCode.NotFound))
    .out(header("Content-Type", "text/html; charset=utf-8"))
    // The SPA shell must always be revalidated: a cached index.html keeps
    // referencing an old hashed bundle, and the app silently runs stale code.
    .out(header("Cache-Control", "no-cache"))
    .out(stringBody)
    .serverLogic[Identity] { (segments: List[String]) =>
      // An unmatched `/api/...` path is a mistake, not a client-side route:
      // answered with the shell, every wrong API call looked like a 200
      // (bugs/16).
      if (segments.headOption.contains("api")) Left(())
      else
        Right(
          Source
            .fromInputStream(
              Thread
                .currentThread()
                .getContextClassLoader
                .getResourceAsStream("index.html"),
              "UTF-8"
            )
            .mkString
        )
    }
)
