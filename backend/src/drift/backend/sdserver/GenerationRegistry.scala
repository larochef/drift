package drift.backend.sdserver

import drift.backend.session.SessionManager
import drift.shared.*

import java.net.URI
import java.net.http.*
import java.util.concurrent.ConcurrentHashMap
import scala.jdk.CollectionConverters.*

/** One generation of this drift run, with the native job it mirrors and the
  * port of the sd-server running it.
  */
final private[sdserver] class GenerationEntry(
    @volatile var generation: Generation,
    val nativeJobId: String,
    val port: Int
)

/** The generations of this drift run, held in memory — runtime state like
  * sessions, gone on restart, rebuilt by the history from the sidecars — and
  * the way to the sessions running them.
  */
final private[sdserver] class GenerationRegistry(
    sessionManager: SessionManager
) {
  private val entries = ConcurrentHashMap[String, GenerationEntry]()

  private val client = HttpClient
    .newBuilder()
    .connectTimeout(java.time.Duration.ofSeconds(3))
    .build()

  def get(generationId: String): Option[GenerationEntry] =
    Option(entries.get(generationId))

  def all: List[GenerationEntry] = entries.values.asScala.toList

  def record(entry: GenerationEntry): Unit =
    entries.put(entry.generation.id, entry)

  def remove(generationId: String): Unit = entries.remove(generationId)

  /** The session, provided it is ready to take a job. */
  def readySession(sessionId: String): Either[String, (Session, Int)] =
    sessionManager.list.find(_.id == sessionId) match {
      case None          => Left(s"session '$sessionId' does not exist")
      case Some(session) =>
        (session.status, session.port) match {
          case (SessionStatus.Ready, Some(port)) => Right((session, port))
          case (SessionStatus.Starting, _)       =>
            Left("the session is still loading its model")
          case _ =>
            Left(s"the session is ${session.status.toString.toLowerCase}")
        }
    }

  def uri(port: Int, path: String): URI =
    URI.create(s"http://127.0.0.1:$port$path")

  def send(request: HttpRequest): HttpResponse[String] =
    client.send(request, HttpResponse.BodyHandlers.ofString())
}
