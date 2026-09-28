package drift.backend.download

import java.net.InetSocketAddress
import java.net.http.HttpClient
import java.nio.file.Files
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean

import com.sun.net.httpserver.HttpServer
import utest.*

/** A connection that goes silent must not hold a download forever, and a cancel
  * must land while it is silent (`StallWatch`).
  */
object DownloaderStallTests extends TestSuite {

  /** Serves a few bytes of a 1 MB body, then says nothing until released. */
  private def silentServer(release: CountDownLatch): HttpServer = {
    val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
    server.createContext(
      "/file",
      exchange => {
        exchange.sendResponseHeaders(200, 1024 * 1024)
        val out = exchange.getResponseBody
        out.write(Array.fill[Byte](1024)(1))
        out.flush()
        release.await()
        exchange.close()
      }
    )
    server.setExecutor(java.util.concurrent.Executors.newCachedThreadPool())
    server.start()
    server
  }

  private def fetch(
      server: HttpServer,
      isCancelled: () => Boolean
  ): (DownloadOutcome, Long) = {
    val dir = Files.createTempDirectory("stall")
    val downloader = Downloader(HttpClient.newHttpClient(), stallMillis = 2000)
    val started = System.currentTimeMillis()
    val outcome = downloader.fetch(
      url = s"http://127.0.0.1:${server.getAddress.getPort}/file",
      partFile = dir.resolve("file.part"),
      finalFile = dir.resolve("file"),
      expectedSha256 = None,
      isCancelled = isCancelled
    )
    (outcome, System.currentTimeMillis() - started)
  }

  val tests = Tests {
    test("a silent connection fails instead of hanging") {
      val release = CountDownLatch(1)
      val server = silentServer(release)
      try {
        val (outcome, took) = fetch(server, () => false)
        assert(outcome.isInstanceOf[DownloadOutcome.Failed], took < 15000)
      } finally { release.countDown(); server.stop(0) }
    }
    test("a cancel lands while the connection is silent") {
      val release = CountDownLatch(1)
      val server = silentServer(release)
      val cancelled = AtomicBoolean(false)
      Thread(() => { Thread.sleep(500); cancelled.set(true) }).start()
      try {
        val (outcome, took) = fetch(server, () => cancelled.get())
        assert(outcome == DownloadOutcome.Cancelled, took < 5000)
      } finally { release.countDown(); server.stop(0) }
    }
  }
}
