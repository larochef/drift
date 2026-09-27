package drift.backend.routes

import drift.backend.session.SessionManager
import drift.shared.*

import java.nio.charset.StandardCharsets
import java.util.concurrent.TimeUnit

import com.github.plokhotnyuk.jsoniter_scala.core.writeToString
import ox.Chunk
import ox.flow.Flow
import sttp.model.StatusCode
import sttp.shared.Identity
import sttp.tapir.*
import sttp.tapir.server.ServerEndpoint
import sttp.tapir.server.netty.sync.OxStreams

/** A session's output as it arrives (`specs/13-log-streaming.md`).
  *
  * One [[LogLine]] as JSON per line, the same NDJSON shape the assistant chat
  * streams - append-only traffic that the status socket's full-topic snapshots
  * are the wrong protocol for. The buffer is replayed first, so opening the
  * view halfway through a load shows what already happened rather than only
  * what happens next.
  *
  * Progress does not come this way: it is small and snapshot-shaped, so it
  * rides the sessions topic of the status socket and reaches every card and
  * panel without anyone opening a log.
  */
def sessionLogEndpoint(
    manager: SessionManager
): ServerEndpoint[OxStreams, Identity] =
  endpoint.get
    .in("api" / "sessions" / path[String]("sessionId") / "logs")
    .errorOut(statusCode(StatusCode.NotFound))
    .out(
      streamTextBody(OxStreams)(
        CodecFormat.TextPlain(),
        Some(StandardCharsets.UTF_8)
      )
    )
    .serverLogic[Identity] { sessionId =>
      manager.logOf(sessionId) match {
        case None      => Left(())
        case Some(log) =>
          val (subscriberId, queue) = log.subscribe()
          val replay = log.snapshot
          Right(
            Flow
              .usingEmit[LogLine] { emit =>
                try {
                  replay.foreach(emit.apply)
                  var running = true
                  while (running) {
                    // A poll rather than a take: the loop has to notice the
                    // client going away, and a session that has stopped
                    // printing must not hold a thread on a blocked read.
                    Option(queue.poll(2, TimeUnit.SECONDS)).foreach(emit.apply)
                    running = manager.logOf(sessionId).isDefined
                  }
                } finally log.unsubscribe(subscriberId)
              }
              .map(line =>
                Chunk.fromArray(
                  (writeToString(line) + "\n").getBytes(StandardCharsets.UTF_8)
                )
              )
          )
      }
    }
