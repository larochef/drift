package drift.backend.nativejobs

import java.net.{InetAddress, InetSocketAddress}
import java.util.concurrent.atomic.AtomicInteger
import scala.concurrent.duration.DurationInt

import com.sun.net.httpserver.HttpServer
import utest.*

/** How a job is followed (`NativeJobClient.follow`): one policy, whoever
  * follows.
  */
object NativeJobClientTests extends TestSuite {

  /** A server whose n-th poll (from 0) of any job is answered by `answer`: a
    * status and a body.
    */
  private def polled[A](answer: Int => (Int, String))(check: Int => A): A = {
    val polls = AtomicInteger(0)
    val server = HttpServer.create(
      InetSocketAddress(InetAddress.getLoopbackAddress, 0),
      0
    )
    server.createContext(
      "/sdcpp/v1/jobs/",
      exchange => {
        val (status, body) = answer(polls.getAndIncrement())
        val bytes = body.getBytes("UTF-8")
        exchange.sendResponseHeaders(status, bytes.length)
        exchange.getResponseBody.write(bytes)
        exchange.close()
      }
    )
    server.start()
    try check(server.getAddress.getPort)
    finally server.stop(0)
  }

  private def job(status: String): String =
    s"""{"id":"a","kind":"img_gen","status":"$status","created":1}"""

  private def follow(port: Int)(seen: String => Unit = _ => ()) =
    NativeJobClient.follow(port, "a", "test", every = 5.millis)(job =>
      seen(job.status)
    )

  val tests = Tests {
    test("a job is followed to its end, each document handed over") {
      val seen = List.newBuilder[String]
      val ended = polled {
        case 0 => 200 -> job("queued")
        case 1 => 200 -> job("generating")
        case _ => 200 -> job("completed")
      }(port => follow(port)(seen += _))
      assert(ended.map(_.map(_.status)) == Right(Some("completed")))
      assert(seen.result() == List("queued", "generating", "completed"))
    }
    test("a job the server forgot ends at once, with that reason") {
      val ended = polled(_ => 404 -> "")(port => follow(port)())
      assert(ended.left.exists(_.contains("no longer knows")))
    }
    test("five failures in a row end it; fewer are forgiven") {
      // an answer that does not read, four times, then the job
      val forgiven = polled {
        case n if n < 4 => 200 -> "not a job"
        case _          => 200 -> job("completed")
      }(port => follow(port)())
      assert(forgiven.isRight)
      val abandoned = polled(_ => 500 -> "")(port => follow(port)())
      assert(abandoned.left.exists(_.contains("HTTP 500")))
    }
    test("a document its follower cannot take counts as a failure") {
      val ended = polled(_ => 200 -> job("generating"))(port =>
        follow(port)(_ => throw RuntimeException("cannot store it"))
      )
      assert(ended.left.exists(_.contains("cannot store it")))
    }
    test("a dead server is the reason, and a follower gone ends it quietly") {
      val dead = polled(_ => 200 -> job("generating"))(port =>
        NativeJobClient.follow(
          port,
          "a",
          "test",
          serverExit = () => Some(137),
          every = 5.millis
        )(_ => ())
      )
      assert(dead.left.exists(_.contains("signal 9")))
      val polls = AtomicInteger(0)
      val gone = polled(_ => 200 -> job("generating"))(port =>
        NativeJobClient.follow(
          port,
          "a",
          "test",
          stillWanted = () => polls.getAndIncrement() < 3,
          every = 5.millis
        )(_ => ())
      )
      assert(gone == Right(None))
    }
  }
}
