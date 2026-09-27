package drift.backend.routes

import drift.shared.*

import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

def fileEndpoints: List[ServerEndpoint[Any, Identity]] = List(
  getHomeDirectory.serverLogicSuccess[Identity](_ =>
    System.getProperty("user.home")
  ),
  listDirectory.serverLogicSuccess[Identity] { path =>
    val dir = Paths.get(path)
    if (Files.isDirectory(dir)) {
      Files
        .list(dir)
        .iterator()
        .asScala
        .toList
        .map { p =>
          val name = p.getFileName.toString
          FileEntry(
            name = name,
            path = p.toAbsolutePath.toString,
            isDirectory = Files.isDirectory(p),
            size = if (Files.isRegularFile(p)) Files.size(p) else 0L,
            extension = name.lastIndexOf('.') match {
              case -1 => ""
              case i  => name.substring(i).toLowerCase
            }
          )
        }
        .sortBy(e => (!e.isDirectory, e.name.toLowerCase))
    } else Nil
  }
)
