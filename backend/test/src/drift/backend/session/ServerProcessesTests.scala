package drift.backend.session

import java.net.{InetAddress, InetSocketAddress, ServerSocket, Socket}

import com.sun.net.httpserver.HttpServer
import utest.*

/** The port test a server launch relies on. */
object ServerProcessesTests extends TestSuite {

  private def listening(): ServerSocket = {
    val server = ServerSocket()
    server.setReuseAddress(true)
    server.bind(InetSocketAddress(InetAddress.getLoopbackAddress, 0))
    server
  }

  val tests = Tests {
    test("a port somebody listens on is taken, and free again once closed") {
      val server = listening()
      val port = server.getLocalPort
      assert(!ServerProcesses.bindable(port))
      server.close()
      assert(ServerProcesses.bindable(port))
    }
    test("a port whose server just closed a connection itself is taken") {
      val server = listening()
      val port = server.getLocalPort
      val client = Socket(InetAddress.getLoopbackAddress, port)
      val accepted = server.accept()
      // The server's side closes first, which leaves its end of the connection
      // held by the kernel for a minute.
      accepted.close()
      assert(client.getInputStream.read() == -1)
      client.close()
      server.close()
      assert(!ServerProcesses.bindable(port))
    }
    test("a server's protocol version is checked where one is expected") {
      def answering(capabilities: String)(check: Int => Unit): Unit = {
        val server = HttpServer.create(
          InetSocketAddress(InetAddress.getLoopbackAddress, 0),
          0
        )
        server.createContext(
          "/sdcpp/v1/capabilities",
          exchange => {
            val bytes = capabilities.getBytes("UTF-8")
            exchange.sendResponseHeaders(200, bytes.length)
            exchange.getResponseBody.write(bytes)
            exchange.close()
          }
        )
        server.start()
        try check(server.getAddress.getPort)
        finally server.stop(0)
      }
      val ours = drift.protocol.JobRoutes.Version
      // the runner of this build
      answering(s"""{"drift":{"protocol":$ours},"samplers":[]}""") { port =>
        assert(ServerProcesses.protocolMismatch(port, runner = true).isEmpty)
      }
      // a runner of another build
      answering(s"""{"drift":{"protocol":${ours + 1}}}""") { port =>
        assert(ServerProcesses.protocolMismatch(port, runner = true).nonEmpty)
        assert(ServerProcesses.protocolMismatch(port, runner = false).nonEmpty)
      }
      // a server that says none: an old runner where one is expected, and
      // sd-server where it is
      answering("""{"samplers":[]}""") { port =>
        assert(ServerProcesses.protocolMismatch(port, runner = true).nonEmpty)
        assert(ServerProcesses.protocolMismatch(port, runner = false).isEmpty)
      }
    }
  }
}
