package drift.backend.postprocess

import drift.backend.postprocess.PostProcessImages.*
import drift.backend.runtime.SdCppBuilds
import drift.shared.*

import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import scala.util.control.NonFatal

/** Upscale by pixel diffusion (`specs/26-tiled-pid.md`): tile by tile on one
  * sd-server, each tile an img_gen job with its quarter of the reference as its
  * only reference image, kept at its own size (`auto_resize_ref_image: false`),
  * and the request's prompt and sampling.
  */
final private[postprocess] class PidUpscale(tiles: TiledJobs) {

  def start(
      date: String,
      fileName: String,
      request: PidUpscaleRequest,
      resuming: Option[String] = None
  ): PostProcessJob =
    tiles.tiledJob(
      "pid",
      date,
      fileName,
      request.runConfigurationId,
      request.runtimeId,
      resuming,
      architecture =>
        Option.unless(architecture.pixelDiffusionDecoder)(
          s"'${architecture.label}' is not a pixel diffusion decoder architecture"
        )
    ) { (src, configuration, _, launch, loras) =>
      for {
        _ <- PidUpscale.refusal(launch.runtime).toLeft(())
        size <- imageSize(src.file).toRight("cannot read the source image")
        target <- PidUpscaleRequest.target(size, request.width, request.height)
        (reference, notes) <- referenceFor(src, target)
      } yield {
        val seed =
          if (request.seed < 0) TiledJobs.drawSeed() else request.seed
        // The decode covers the padded reference ×4; the result is cropped
        // back to the target. The browser lays out the same tiles to price the
        // job (`PidUpscaleRequest.tilesFor`).
        val rows = PidUpscaleRequest.tilesFor(
          target,
          PidUpscaleRequest.maxTileFor(launch.runtime)
        )
        tiles.startTiles(
          "pid",
          src,
          rows,
          PausedWork.Pid(request.copy(seed = seed)),
          resuming,
          // A PiD tile is in target pixels; the picture on screen is the
          // source, four times smaller at a true ×4.
          rows.flatten.map { tile =>
            val scale = size._1.toDouble / target._1
            ImageRegion(
              math.round(tile.x * scale).toInt,
              math.round(tile.y * scale).toInt,
              math.round(tile.width * scale).toInt,
              math.round(tile.height * scale).toInt
            )
          }
        )(job =>
          tiles.runTiles(
            job,
            src,
            launch,
            reference,
            scale = 4,
            runConfigurationId = configuration.id,
            // The configuration's sampler, scheduler and VAE tiling, as the
            // server's defaults carry them; the request's steps and cfg.
            request = TileRequests.Images((defaults, _, input) =>
              Right(
                ImageGenerationParameters(
                  prompt = request.prompt,
                  negativePrompt = request.negativePrompt,
                  clipSkip = defaults.clipSkip,
                  width = input.window.width,
                  height = input.window.height,
                  seed = seed,
                  refImages = List(input.image),
                  autoResizeRefImage = Some(false),
                  lora = loras.selections,
                  sampleParams = defaults.sampleParams.copy(
                    sampleSteps = request.steps,
                    guidance = defaults.sampleParams.guidance
                      .copy(txtCfg = request.cfgScale)
                  ),
                  vaeTilingParams = defaults.vaeTilingParams
                )
              )
            ),
            target = target,
            rows = rows,
            notes = notes ++ TiledJobs.loraNote(loras.selections),
            derivation = Derivation(
              parentId = src.parent.id,
              parentDate = src.date,
              parentFileName = src.fileName,
              operation = "pid",
              width = Some(target._1),
              height = Some(target._2),
              configurationId = Some(configuration.id),
              prompt = Some(request.prompt).filter(_.nonEmpty),
              steps = Some(request.steps),
              seed = Some(seed)
            ),
            // PiD drifts in colour, 4 to 7 levels on average and up to 34 on
            // an illustration (bug 39): the tile's own crop of the reference,
            // at the tile's size, carries the colours to keep
            correctTile = Some((decoded, crop) =>
              colourMatched(
                decoded,
                scaledCopy(crop, decoded.getWidth, decoded.getHeight)
              )
            ),
            resumed = resuming.isDefined
          )
        )
      }
    }

  /** The reference: the source at a quarter of the target — the ×4 the decoders
    * are trained for; sd-cpp does not check the ratio (a 1024² source asked for
    * 2048² re-lights the scene) — which is the source itself when it already is
    * that size, else a `fill` scaled copy; then padded to multiples of 16 by
    * repeating its last row and column, since a PiD input must be.
    */
  private def referenceFor(
      src: PostProcessSource,
      target: (Int, Int)
  ): Either[String, (BufferedImage, List[String])] = {
    val (width, height) = (target._1 / 4, target._2 / 4)
    try
      Option(ImageIO.read(src.file.toFile)) match {
        case None        => Left("cannot decode the source image")
        case Some(image) =>
          val content =
            if (image.getWidth == width && image.getHeight == height) image
            else fill(image, width, height)
          val (paddedWidth, paddedHeight) =
            PidUpscaleRequest.referenceSizeOf(target)
          val reference = padded(content, paddedWidth, paddedHeight)
          Right(
            (
              reference,
              List(
                s"source ${image.getWidth}x${image.getHeight}, reference ${width}x$height padded to ${reference.getWidth}x${reference.getHeight}, cropped back to ${target._1}x${target._2} after the decode"
              )
            )
          )
      }
    catch {
      case NonFatal(e) => Left(s"cannot read the source image: ${e.getMessage}")
    }
  }
}

private[postprocess] object PidUpscale {

  /** Why `runtime` cannot run a PiD job, if it cannot: its sd-server scales
    * every reference to the tile's size (`SdCppBuilds.keepsReferenceSize`), and
    * PiD, conditioned on that blurred reference instead of the small crop,
    * returns a smear (bugs/25).
    */
  def refusal(runtime: Runtime): Option[String] =
    Option.unless(SdCppBuilds.keepsReferenceSize(runtime))(
      s"PiD needs sd-cpp master-${SdCppBuilds.FirstKeepingReferenceSize} " +
        "or newer, whose sd-server keeps a reference image its own size; " +
        s"'${runtime.label}' is ${runtime.releaseTag} — update it"
    )
}
