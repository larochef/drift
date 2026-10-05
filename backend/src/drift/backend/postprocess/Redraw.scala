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
  * steps, and those steps all run whatever the strength; the instructions are
  * appended to the built-in restoration prompt. Each tile is painted beside the
  * 3×3 block of tiles around it, and comes back with the source's colour
  * (`specs/45`).
  *
  * A request naming a region repaints that part alone: the tiles cover a window
  * grown around the selection to something the model paints well, and the
  * result is feathered back into the untouched source. Everything else — the
  * restoration prompt, the instructions, the reference, the tiling — is what a
  * full redraw does.
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
          // A tile that sees nothing of its surroundings is repainted as
          // something else: an areola erased, sand turned to rock (`specs/45`).
          .orElse(
            Option.unless(
              architecture.referenceImages == ReferenceImageUse.Context
            )(
              s"'${architecture.label}' does not take a reference image as " +
                "context, and a redraw needs one: without it each tile is " +
                "repainted as something else"
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
        laidOut = TiledArea.of(
          image,
          region,
          request.tileSize,
          request.minimumWindowSide,
          request.selectionMargin,
          request.gridOffsetX,
          request.gridOffsetY,
          architecture.sizeMultiple
        )
        settings <- Redraw.settingsFor(request, region, laidOut.rows.flatten)
      } yield {
        val source = (image.getWidth, image.getHeight)
        val TiledArea.Area(area, reference, rows) = laidOut
        val target = (area.width, area.height)
        // Only a model that takes a reference as context gets here, so one is
        // sent unless the job weighed the trade itself and asked for none.
        val useReference = request.useReference.getOrElse(true)
        val seed =
          if (request.seed < 0) TiledJobs.drawSeed() else request.seed
        // The model sees the picture, so it is told the job rather than the
        // subject: the source's prompt is not sent.
        // A tile the assistant read is told its own materials after the
        // template, and before the words written for the whole job.
        def instructionsFor(tile: Tiling.Tile): String = List(
          restoration,
          settings.get(tile).fold("")(_.prompt.trim),
          request.instructions.trim
        ).filter(_.nonEmpty).mkString(" ")
        def strengthOf(tile: Tiling.Tile): Double =
          settings.get(tile).fold(request.strength)(_.strength)
        // Each tile's reference is the 3×3 block of tiles around it, cut
        // from the source and fitted within the reference size: the whole
        // picture squeezed into it left a 16k tile a few dozen pixels, which
        // told the model nothing (`specs/45`).
        def neighbourhoodOf(window: Tiling.Tile): Tiling.Tile =
          Redraw.neighbourhood(
            window.copy(x = window.x + area.x, y = window.y + area.y),
            source
          )
        // An sd-server before `SdCppBuilds.FirstKeepingReferenceSize` stretches every
        // reference to the tile's width × height, so a portrait beside a
        // square tile came out half again as wide (François, 2026-09-22): on
        // such a build the reference is letterboxed to each tile's shape and
        // scaled evenly; a later one is told to keep it as it is.
        val keepsReference = SdCppBuilds.keepsReferenceSize(launch.runtime)
        def referenceFor(window: Tiling.Tile): String = {
          val block = neighbourhoodOf(window)
          val fitted = fitWithin(
            image.getSubimage(block.x, block.y, block.width, block.height),
            request.contextSide
          )
          if (keepsReference) dataUrl(fitted)
          else dataUrl(letterboxed(fitted, window.width, window.height))
        }
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
          if (request.keepTiles)
            Files.createDirectories(jobs.files.tilesDirOf(job))
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
            request = TileRequests.Images((defaults, features, input) =>
              Right(
                ImageGenerationParameters(
                  prompt = {
                    val placed = input.window.copy(
                      x = input.window.x + area.x,
                      y = input.window.y + area.y
                    )
                    val block = neighbourhoodOf(input.window)
                    val framing =
                      if (useReference)
                        Redraw.framing(
                          placed.copy(
                            x = placed.x - block.x,
                            y = placed.y - block.y
                          ),
                          (block.width, block.height),
                          withReference = true
                        )
                      else Redraw.framing(placed, source, withReference = false)
                    s"$framing ${instructionsFor(input.tile)}"
                  },
                  negativePrompt = request.negativePrompt,
                  clipSkip = defaults.clipSkip,
                  width = input.window.width,
                  height = input.window.height,
                  strength = strengthOf(input.tile),
                  seed = seed,
                  initImage = Some(input.image),
                  // the tile's own mask, with context around it; else the
                  // selection's, for a repair that must leave the rest alone
                  maskImage = input.mask.orElse(
                    region
                      .filter(_ =>
                        request.maskSelection &&
                          features.getOrElse("mask_image", false)
                      )
                      .map(selection =>
                        dataUrl(
                          Redraw.selectionMask(
                            selection.copy(
                              x = selection.x - area.x - input.window.x,
                              y = selection.y - area.y - input.window.y
                            ),
                            input.window.width,
                            input.window.height,
                            request.selectionMargin
                          )
                        )
                      )
                  ),
                  refImages =
                    if (useReference) List(referenceFor(input.window))
                    else List.empty,
                  autoResizeRefImage =
                    Option.when(useReference && keepsReference)(false),
                  lora = loras.selections,
                  // the session's sampling under the configuration's LoRAs'
                  // (`specs/49`), under the steps the request asks for
                  sampleParams = {
                    val sampling = loras.sampling.over(defaults.sampleParams)
                    val steps = request.steps.getOrElse(sampling.sampleSteps)
                    // A custom schedule sets its own steps.
                    if (sampling.customSigmas.nonEmpty) sampling
                    else
                      sampling.copy(sampleSteps =
                        Redraw.scheduledSteps(steps, strengthOf(input.tile))
                      )
                  },
                  vaeTilingParams = defaults.vaeTilingParams
                )
              )
            ),
            target = target,
            rows = rows,
            notes = List(
              region.fold(s"source ${source._1}x${source._2}")(selection =>
                s"selection ${selection.width}x${selection.height} at ${selection.x},${selection.y} of ${source._1}x${source._2}, repainted through a ${area.width}x${area.height} window at ${area.x},${area.y} and feathered back into the source"
              ) + s", padded to ${reference.getWidth}x${reference.getHeight}, cropped back after the redraw; strength ${request.strength}; " +
                (if (useReference)
                   s"each tile beside the 3×3 block of tiles around it as reference, within ${request.contextSide} px" +
                     (if (keepsReference) ", kept at that size"
                      else
                        ", letterboxed to each tile's shape (this sd-server stretches a reference to the tile)")
                 else "no reference image (switched off for this job)") +
                request.steps.fold("")(steps =>
                  s"; $steps steps run, ${Redraw.scheduledSteps(steps, request.strength)} scheduled"
                ) + (if (request.matchColour)
                       "; each tile given the source's colour back"
                     else "; tiles keep the colour they were painted with") +
                (if (request.maskSelection && region.isDefined)
                   "; the selection repainted alone under a mask, where the server takes one"
                 else "") +
                (if (settings.nonEmpty)
                   s"; each tile at its own strength and with its own prompt, as the assistant read the picture — ${settings.values
                       .count(_.strength <= 0)} of ${settings.size} left as they are"
                 else "") +
                (if (request.softenRadius > 0)
                   s"; tiles softened by ${request.softenRadius} px first"
                 else "") +
                (if (request.contextMargin > 0)
                   s"; each tile seen with up to ${request.contextMargin} px of the picture as redrawn so far around it, kept as it is under a mask, tiles in order"
                 else "")
            ) ++ TiledJobs.loraNote(loras.selections),
            prepareTile =
              if (request.softenRadius > 0)
                PostProcessImages.softened(_, request.softenRadius)
              else identity,
            correctTile =
              Option.when(request.matchColour)(PostProcessImages.colourMatched),
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
              instructions = Some(request.instructions.trim).filter(_.nonEmpty),
              planned = Option.when(settings.nonEmpty)(true)
            ),
            finish = region.fold[PictureFinish](PictureFinish.AsPainted)(
              PictureFinish.PastedInto(image, area, _)
            ),
            keepTiles = request.keepTiles,
            resumed = resuming.isDefined,
            untouched = tile => settings.get(tile).exists(_.strength <= 0),
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

  /** The mask of `selection` (in a window's own pixels) inside a `width` ×
    * `height` window: white over the selection grown by half the `margin` the
    * paste feathers across, softened by as much, black around it.
    */
  def selectionMask(
      selection: ImageRegion,
      width: Int,
      height: Int,
      margin: Int
  ): java.awt.image.BufferedImage = {
    val mask = java.awt.image.BufferedImage(
      width,
      height,
      java.awt.image.BufferedImage.TYPE_INT_RGB
    )
    val grown = margin / 2
    val graphics = mask.createGraphics()
    try {
      graphics.setColor(java.awt.Color.WHITE)
      graphics.fillRect(
        selection.x - grown,
        selection.y - grown,
        selection.width + 2 * grown,
        selection.height + 2 * grown
      )
    } finally graphics.dispose()
    PostProcessImages.softened(mask, grown / 2.0)
  }

  /** The request's own settings for each of `tiles` (`specs/52`), by the tile's
    * place — none when the request carries none. They were read off the whole
    * picture for one layout: a selection, or tiles cut elsewhere, is refused
    * rather than painted with another tile's prompt.
    */
  def settingsFor(
      request: RedrawRequest,
      region: Option[ImageRegion],
      tiles: List[Tiling.Tile]
  ): Either[String, Map[Tiling.Tile, TileSettings]] =
    if (request.tiles.isEmpty) Right(Map.empty)
    else if (region.isDefined)
      Left(
        "the assistant's settings are for the whole picture: a selection is " +
          "redrawn without them"
      )
    else {
      val byPlace = request.tiles
        .map(settings =>
          Tiling.Tile(
            settings.region.x,
            settings.region.y,
            settings.region.width,
            settings.region.height
          ) -> settings
        )
        .toMap
      val missing = tiles.filterNot(byPlace.contains)
      if (missing.nonEmpty || byPlace.size != tiles.size)
        Left(
          "the assistant's settings were made for other tiles than this " +
            "job cuts — the tile size or the grid moved since: ask again"
        )
      else if (byPlace.values.exists(s => s.strength < 0 || s.strength > 1))
        Left("a tile's strength must be from 0 to 1")
      else if (byPlace.values.forall(_.strength <= 0))
        Left("every tile is left as it is: nothing to redraw")
      else Right(byPlace)
    }

  /** The 3×3 block of tiles around `tile`, in the picture's pixels: three tiles
    * a side less the two overlaps between them, centred on the tile, shifted to
    * stay inside the picture and no larger than it (`specs/45`). A 1280 tile's
    * block is 3328 px, so the tile is about a third of its reference whatever
    * the picture's size; a picture no larger than that is its own block.
    */
  def neighbourhood(tile: Tiling.Tile, size: (Int, Int)): Tiling.Tile = {
    def axis(start: Int, length: Int, total: Int): (Int, Int) = {
      val side = (3 * length - 2 * Tiling.Overlap).max(length).min(total)
      val centred = start + length / 2 - side / 2
      (centred.max(0).min(total - side), side)
    }
    val (x, width) = axis(tile.x, tile.width, size._1)
    val (y, height) = axis(tile.y, tile.height, size._2)
    Tiling.Tile(x, y, width, height)
  }

  /** How many steps to schedule so that `steps` of them run at `strength`:
    * img2img runs only the last ⌊scheduled × strength⌋, sd-cpp and the runner
    * alike, in single precision as they compute it. Asking for `steps` at 0.4
    * on a 4-step model ran one, which repaints nothing (`specs/45`, Part 1).
    */
  def scheduledSteps(steps: Int, strength: Double): Int =
    if (strength >= 1) steps
    else
      Iterator
        .from(steps)
        .find(scheduled => (scheduled * strength.toFloat).toInt >= steps)
        .get

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
