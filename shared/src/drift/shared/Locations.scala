package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Where this drift keeps its files, each as the page shows it: the folder as
  * the backend resolved it at startup (`docs/files-and-folders.md`), with the
  * home directory written `~`. What the page says of a folder it reads here: a
  * relocated drift names the folders it really uses.
  */
case class FileLocations(
    /** Entities, one JSON file each. */
    configuration: String,
    /** The tokens of the sites, in the configuration. */
    authTokens: String,
    /** Models drift downloaded or converted itself. */
    models: String,
    modelScope: String,
    loras: String,
    upscalers: String,
    runtimes: String,
    /** Generated images and videos. */
    outputs: String,
    /** The HuggingFace hub cache, shared with every other HF tool. */
    huggingFace: String
)
object FileLocations {
  given JsonValueCodec[FileLocations] = JsonCodecMaker.make
}

lazy val getLocations: PublicEndpoint[Unit, Unit, FileLocations, Any] =
  base.get.in("locations").out(jsonBody[FileLocations])
