package drift.backend.postprocess

import drift.backend.postprocess.PostProcessImages.*
import drift.backend.runtime.SdCppBuilds
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.Files
import javax.imageio.ImageIO

/** Redraw (`specs/27-redraw.md`): the model of an `image` run configuration
  * repaints the source at `strength`, one img2img job per tile on one
  * sd-server, the tiles blended back at the source's size. The server's
  * defaults — the configuration's steps and cfg — hold unless the request names
  * steps; the instructions are appended to the built-in restoration prompt.
  *
  * A request naming a region repaints that part alone: the tiles cover a window
  * grown around the selection to something the model paints well, and the
  * result is feathered back into the untouched source. Everything else — the
  * restoration prompt, the instructions, the whole image as reference, the
  * tiling — is what a full redraw does.
  */
final private[postprocess] class Redraw(
    jobs: PostProcessJobs,
    tiles: TiledJobs,
    storage: StorageService
) {

  def start(
      date: String,
      fileName: String,
      request: RedrawRequest,
      resuming: Option[String] = None
  ): PostProcessJob =
    tiles.tiledJob(
      "redraw",
      date,
      fileName,
      request.runConfigurationId,
      request.runtimeId,
      resuming,
      architecture =>
        Option
          .unless(
            architecture.tool == RuntimeTool.SdCpp &&
              architecture.tags.contains("image")
          )(s"'${architecture.label}' is not an image architecture")
          .orElse(
            Option.unless(architecture.initImage)(
              s"'${architecture.label}' does not repaint an image it is " +
                "given, so it cannot redraw one"
            )
          )
    ) { (src, configuration, architecture, launch, loras) =>
      for {
        _ <- Either.cond(
          request.strength > 0 && request.strength <= 1,
          (),
          "strength must be above 0 and at most 1"
        )
        _ <- Either.cond(
          request.steps.forall(_ >= 1),
          (),
          "steps must be at least 1"
        )
        _ <- TiledArea.checkArea(
          request.tileSize,
          request.minimumWindowSide,
          request.selectionMargin
        )
        _ <- Either.cond(
          request.contextSide >= 256 && request.contextSide <= 2048,
          (),
          "the reference size must be from 256 to 2048 px"
        )
        _ <- Either.cond(
          request.softenRadius >= 0 && request.softenRadius <= 8,
          (),
          "the soften radius must be from 0 to 8 px"
        )
        _ <- Either.cond(
          request.contextMargin >= 0 && request.contextMargin <= 512,
          (),
          "the context around a tile must be from 0 to 512 px"
        )
        _ <- Either.cond(
          request.tileSize + 2 * request.contextMargin <= 2048,
          (),
          "a tile with its context on both sides must stay within 2048 px — " +
            "a smaller tile or less context"
        )
        restoration <- TiledArea.templateTextOf(
          storage,
          request.templateId,
          PromptKind.RedrawRestoration,
          PromptTemplate.DefaultRedrawId
        )
        image <- Option(ImageIO.read(src.file.toFile))
          .toRight("cannot decode the source image")
        region <- TiledArea.regionOf(
          request.region,
          image.getWidth,
          image.getHeight
        )
      } yield {
        val source = (image.getWidth, image.getHeight)
        val TiledArea.Area(area, reference, rows) = TiledArea.of(
          image,
          region,
          request.tileSize,
          request.minimumWindowSide,
          request.selectionMargin,
          request.gridOffsetX,
          request.gridOffsetY,
          architecture.sizeMultiple
        )
        val target = (area.width, area.height)
        // What the architecture says the model does with a reference image,
        // unless the request has weighed the trade itself. A model whose
        // preset hands the reference to the diffusion model is given none:
        // Krea2, handed the whole picture beside a tile, returned mosaic
        // corruption over the face (François, 2026-09-19), and sd-server
        // exposes no `ref_image_args` to soften the preset.
        val useReference = request.useReference match {
          case Some(asked) => asked
          case None        =>
            architecture.referenceImages == ReferenceImageUse.Context
        }
        // Why none was sent, which is not one reason but three: the job asked
        // for none, the family has no preset at all, or its preset is an
        // editing one and drift withholds the reference on purpose.
        val noReferenceReason =
          if (request.useReference.contains(false)) "switched off for this job"
          else
            architecture.referenceImages match {
              case ReferenceImageUse.Edit =>
                s"sd-cpp hands '${architecture.label}' a reference as the " +
                  "image to edit, which would paint the whole picture into " +
                  "every tile"
              case _ =>
                s"'${architecture.label}' has no reference preset in sd-cpp"
            }
        val seed =
          if (request.seed < 0) TiledJobs.drawSeed() else request.seed
        // The model sees the picture, so it is told the job rather than the
        // subject: the source's prompt is not sent.
        val instructions = List(
          restoration,
          request.instructions.trim
        ).filter(_.nonEmpty).mkString(" ")
        // The whole source, downscaled once, beside every tile — built only
        // when it is going to be sent.
        lazy val whole = fitWithin(image, request.contextSide)
        lazy val wholeImage = dataUrl(whole)
        // An sd-server before `SdCppBuilds.FirstKeepingReferenceSize` stretches every
        // reference to the tile's width × height, so a portrait beside a
        // square tile came out half again as wide (François, 2026-09-22): on
        // such a build the reference is letterboxed to each tile's shape and
        // scaled evenly; a later one is told to keep it as it is.
        val keepsReference = SdCppBuilds.keepsReferenceSize(launch.runtime)
        def referenceFor(tile: Tiling.Tile): String =
          if (keepsReference) wholeImage
          else dataUrl(letterboxed(whole, tile.width, tile.height))
        tiles.startTiles(
          "redraw",
          src,
          rows,
          PausedWork.Redraw(request.copy(seed = seed)),
          resuming,
          // The tiles cover the window, which sits at `area` in the picture.
          rows.flatten.map(tile =>
            tile.copy(x = tile.x + area.x, y = tile.y + area.y)
          )
        ) { job =>
          if (request.keepTiles) {
            Files.createDirectories(jobs.files.tilesDirOf(job))
            if (useReference)
              ImageIO.write(
                whole,
                "png",
                jobs.files.tilesDirOf(job).resolve("reference.png").toFile
              )
          }
          tiles.runTiles(
            job,
            src,
            launch,
            reference,
            scale = 1,
            // No feature is consulted: sd-server answers `ref_images: true`
            // for every model from a hardcoded table, so it cannot say which
            // ones mean it (`specs/27-redraw.md`).
            runConfigurationId = configuration.id,
            // What the model paints is the window: the tile, or the tile with
            // its context around it and a mask keeping the context as it is.
            request = (defaults, _, input) =>
              Right(
                ImageGenerationParameters(
                  prompt = {
                    val placed = input.window.copy(
                      x = input.window.x + area.x,
                      y = input.window.y + area.y
                    )
                    s"${Redraw.framing(placed, source, useReference)} " +
                      instructions
                  },
                  negativePrompt = request.negativePrompt,
                  clipSkip = defaults.clipSkip,
                  width = input.window.width,
                  height = input.window.height,
                  strength = request.strength,
                  seed = seed,
                  initImage = Some(input.image),
                  maskImage = input.mask,
                  refImages =
                    if (useReference) List(referenceFor(input.window))
                    else List.empty,
                  autoResizeRefImage =
                    Option.when(useReference && keepsReference)(false),
                  lora = loras,
                  sampleParams = request.steps.fold(defaults.sampleParams)(
                    steps => defaults.sampleParams.copy(sampleSteps = steps)
                  ),
                  vaeTilingParams = defaults.vaeTilingParams
                )
              ),
            target = target,
            rows = rows,
            notes = List(
              region.fold(s"source ${source._1}x${source._2}")(selection =>
                s"selection ${selection.width}x${selection.height} at ${selection.x},${selection.y} of ${source._1}x${source._2}, repainted through a ${area.width}x${area.height} window at ${area.x},${area.y} and feathered back into the source"
              ) + s", padded to ${reference.getWidth}x${reference.getHeight}, cropped back after the redraw; strength ${request.strength}; " +
                (if (useReference)
                   s"the whole image as reference at ${whole.getWidth}x${whole.getHeight}" +
                     (if (keepsReference) ", kept at that size"
                      else
                        ", letterboxed to each tile's shape (this sd-server stretches a reference to the tile)")
                 else s"no reference image ($noReferenceReason)") +
                (if (request.softenRadius > 0)
                   s"; tiles softened by ${request.softenRadius} px first"
                 else "") +
                (if (request.contextMargin > 0)
                   s"; each tile seen with up to ${request.contextMargin} px of the picture as redrawn so far around it, kept as it is under a mask, tiles in order"
                 else "")
            ) ++ TiledJobs.loraNote(loras),
            prepareTile =
              if (request.softenRadius > 0)
                PostProcessImages.softened(_, request.softenRadius)
              else identity,
            derivation = Derivation(
              parentId = src.parent.id,
              parentDate = src.date,
              parentFileName = src.fileName,
              operation = "redraw",
              width = Some(source._1),
              height = Some(source._2),
              region = region,
              configurationId = Some(configuration.id),
              steps = request.steps,
              seed = Some(seed),
              strength = Some(request.strength),
              instructions = Some(request.instructions.trim).filter(_.nonEmpty)
            ),
            finish = region.fold[PictureFinish](PictureFinish.AsPainted)(
              PictureFinish.PastedInto(image, area, _)
            ),
            keepTiles = request.keepTiles,
            resumed = resuming.isDefined,
            context = Option.when(request.contextMargin > 0)(
              TileWindow.TileContext(
                request.contextMargin,
                architecture.sizeMultiple
              )
            )
          )
        }
      }
    }
}

private[postprocess] object Redraw {

  /** What drift puts in front of the restoration prompt. It differs by what the
    * model was actually handed, which is the whole point: a redraw with no
    * reference repainted the entire scene into a single tile (François,
    * 2026-09-19).
    *
    * With a reference, the tile is placed inside it and the reference is named
    * for what it is — context, identity included — since the model sees both
    * images and must be told which one it is repainting.
    *
    * Without one, the crop is all there is. Naming "a larger photograph" and
    * quoting percentages of a frame the model cannot see describes a scene it
    * can only imagine, and imagining it is precisely what went wrong; so the
    * sentence says the opposite, about the only image in the request.
    */
  def framing(
      tile: Tiling.Tile,
      size: (Int, Int),
      withReference: Boolean
  ): String =
    if (withReference)
      PostProcessImages.position(tile, size) +
        " The reference image is context only: keep every person exactly as" +
        " they appear in it — eye colour, skin tone, hair colour and style," +
        " facial features and ethnicity — and repaint only what this crop" +
        " already shows."
    else
      "This image is a close crop of a photograph. Repaint only what it" +
        " already shows, at the same framing and scale: do not widen the" +
        " view, do not add anything that is not already visible, and keep" +
        " every element exactly where it is."
}
