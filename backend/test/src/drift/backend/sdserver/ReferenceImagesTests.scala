package drift.backend.sdserver

import drift.backend.postprocess.PostProcessImages
import drift.shared.*
import utest.*

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.util.Base64
import javax.imageio.ImageIO

/** How a generation's reference images are handed to sd-server.
  *
  * Worth testing because a wrong answer is silent: a stretched reference is
  * still an image the model accepts, it just draws the subject half again as
  * wide.
  */
object ReferenceImagesTests extends TestSuite {

  private def architecture(id: String) = Architecture(
    id = id,
    label = id,
    tool = RuntimeTool.SdCpp,
    checkpoints = List.empty,
    defaultParameters = Map.empty,
    sizeMultiple = 16,
    modelKind = None,
    runners = List(RuntimeEngine.SdCpp)
  )

  private def runtime(tag: String) = Runtime(
    id = "latest-rocm",
    label = "latest",
    tool = RuntimeTool.SdCpp,
    backend = RuntimeBackend.Rocm,
    releaseTag = tag,
    installedAt = "/nowhere",
    modelKinds = None
  )

  private def portrait: String = {
    val image = BufferedImage(512, 768, BufferedImage.TYPE_INT_RGB)
    PostProcessImages.dataUrl(image)
  }

  private def sizeOf(dataUrl: String): (Int, Int) = {
    val image = ImageIO.read(
      ByteArrayInputStream(
        Base64.getMimeDecoder.decode(dataUrl.drop(dataUrl.indexOf(',') + 1))
      )
    )
    (image.getWidth, image.getHeight)
  }

  private val request = ImageGenerationParameters(
    prompt = "",
    width = 1024,
    height = 1024,
    refImages = List(portrait)
  )

  val tests = Tests {

    test("sized as sd-cpp's resize before the VAE would") {
      val klein = architecture("flux.2-klein-9B")
      // A megapixel, the portrait's shape, sides on 16.
      assert(
        ReferenceImages.vaeSizeOf(1024, 1536, 1024, 1024, klein) == (832, 1248)
      )
      // Never more than the output's area.
      assert(
        ReferenceImages.vaeSizeOf(1024, 1536, 512, 512, klein) == (416, 624)
      )
      // Qwen-Image 2.1 takes the output's area, on 32.
      assert(
        ReferenceImages.vaeSizeOf(
          1024,
          1536,
          1328,
          1328,
          architecture("qwen-image-2.1")
        ) == (1088, 1632)
      )
    }

    test("a build that keeps sizes gets the resized reference and false") {
      val sent = ReferenceImages.prepared(
        request,
        architecture("flux.2-klein-9B"),
        runtime("master-892-78557f8")
      )
      assert(sent.autoResizeRefImage == Some(false))
      assert(sizeOf(sent.refImages.head) == (832, 1248))
    }

    test(
      "an older build gets the reference letterboxed to the output's shape"
    ) {
      val sent = ReferenceImages.prepared(
        request,
        architecture("flux.2-klein-9B"),
        runtime("master-890-74988b2")
      )
      assert(sent.autoResizeRefImage.isEmpty)
      assert(sizeOf(sent.refImages.head) == (768, 768))
    }

    test("no reference, nothing changed") {
      val plain = request.copy(refImages = List.empty)
      assert(
        ReferenceImages.prepared(
          plain,
          architecture("flux.2-klein-9B"),
          runtime("master-892-78557f8")
        ) == plain
      )
    }
  }
}
