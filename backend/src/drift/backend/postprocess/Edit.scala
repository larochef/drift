package drift.backend.postprocess

import drift.backend.runtime.LaunchRuntime
import drift.backend.storage.StorageService
import drift.shared.*

import javax.imageio.ImageIO

/** Edit (`specs/39-seamless-edit.md`): an instruction says what should be
  * different, and a model that edits by instruction makes the change tile by
  * tile while the rest of the picture stays what it was.
  *
  * Each tile goes to the model as the image to edit — its only reference, no
  * init image — with the edit template and the instruction. The tiles run in
  * order, each cut from the picture as edited so far, and each comes back
  * through `EditComposite`: the source's pixels wherever the model changed
  * nothing, the change brought to the source's colours. A request naming a
  * region changes that part alone, through a window grown around it and
  * feathered back into the source, as a redraw's is.
  */
final private[postprocess] class Edit(
    jobs: PostProcessJobs,
    tiles: TiledJobs,
    storage: StorageService,
    carry: EditCarry
) {

  def start(
      date: String,
      fileName: String,
      request: EditRequest,
      resuming: Option[String] = None
  ): PostProcessJob =
    tiles.tiledJob(
      "edit",
      date,
      fileName,
      request.runConfigurationId,
      request.runtimeId,
      resuming,
      architecture =>
        Option.unless(
          architecture.tool == RuntimeTool.SdCpp &&
            architecture.tags.contains(ArchitectureTags.Edit)
        )(
          s"'${architecture.label}' is not tagged ${ArchitectureTags.Edit}: " +
            "only a model that edits an image by instruction keeps what the " +
            "instruction does not mention"
        )
    ) { (src, configuration, architecture, launch, loras) =>
      val instruction = request.instructions.trim
      for {
        _ <- Either.cond(
          instruction.nonEmpty,
          (),
          "an edit needs an instruction: what should be different"
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
        template <- TiledArea.templateTextOf(
          storage,
          request.templateId,
          PromptKind.Edit,
          PromptTemplate.DefaultEditId
        )
        image <- Option(ImageIO.read(src.file.toFile))
          .toRight("cannot decode the source image")
        region <- TiledArea.regionOf(
          request.region,
          image.getWidth,
          image.getHeight
        )
        // The upscaler an edit is carried up by: a SeedVR2 configuration, on
        // the drift runner — the only engine with its `upscale` job.
        upscaler <- request.upscaleConfigurationId.fold(
          Right(None): Either[String, Option[(RunConfiguration, LaunchRuntime)]]
        )(id =>
          tiles
            .resolved(
              id,
              upscaling =>
                Option.unless(SeedVr2UpscaleRequest.runs(upscaling))(
                  s"'${upscaling.label}' is not a SeedVR2 architecture: an edit is carried up by a SeedVR2 upscaler"
                )
            )
            .flatMap((upscaling, runtime) =>
              Either.cond(
                runtime.runtime.engine == RuntimeEngine.DriftRunner,
                Some((upscaling, runtime)),
                s"SeedVR2 runs on the drift runner only, and '${upscaling.label}' is set to ${runtime.runtime.engine}"
              )
            )
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
        val seed =
          if (request.seed < 0) TiledJobs.drawSeed() else request.seed
        val prompt =
          List(template.trim, instruction).filter(_.nonEmpty).mkString(" ")
        val derivation = Derivation(
          parentId = src.parent.id,
          parentDate = src.date,
          parentFileName = src.fileName,
          operation = "edit",
          width = Some(source._1),
          height = Some(source._2),
          region = region,
          configurationId = Some(configuration.id),
          steps = request.steps,
          seed = Some(seed),
          instructions = Some(instruction)
        )
        // Made once and carried up: not a tiled job, so it is not paused — it
        // is one pass of the model and a few of the upscaler.
        def carried(
            upscaling: RunConfiguration,
            runtime: LaunchRuntime
        ): PostProcessJob = {
          // What the model sees around a selection is wider than a tiled
          // edit's window: it is one pass whatever its size, and the model
          // needs the picture around the thing to know what it is.
          val seen = region.fold(area)(selection =>
            Tiling.window(
              selection,
              image.getWidth,
              image.getHeight,
              EditRequest
                .contextSide(selection)
                .max(request.minimumWindowSide),
              request.selectionMargin,
              multiple = architecture.sizeMultiple
            )
          )
          jobs.start("edit", src, tiles = List(seen))(job =>
            carry.run(
              job,
              carry.Work(
                src,
                image,
                seen,
                region,
                prompt,
                request.copy(seed = seed),
                configuration,
                launch,
                loras,
                architecture.sizeMultiple,
                upscaling,
                runtime,
                derivation
              )
            )
          )
        }
        def tiled: PostProcessJob = tiles.startTiles(
          "edit",
          src,
          rows,
          PausedWork.Edit(request.copy(seed = seed)),
          resuming,
          rows.flatten.map(tile =>
            tile.copy(x = tile.x + area.x, y = tile.y + area.y)
          )
        ) { job =>
          tiles.runTiles(
            job,
            src,
            launch,
            reference,
            scale = 1,
            runConfigurationId = configuration.id,
            // No framing sentence: the tile is the one image the model is
            // given, and it is the one to change. No init image either — at
            // any strength img2img either ignores the instruction or relights
            // what it should keep (`specs/39`).
            request = TileRequests.Images((defaults, _, input) =>
              Right(
                ImageGenerationParameters(
                  prompt = prompt,
                  clipSkip = defaults.clipSkip,
                  width = input.window.width,
                  height = input.window.height,
                  seed = seed,
                  refImages = List(input.image),
                  lora = loras.selections,
                  // the session's sampling under the configuration's LoRAs'
                  // (`specs/49`), under the steps the request asks for
                  sampleParams = {
                    val sampling = loras.sampling.over(defaults.sampleParams)
                    request.steps.fold(sampling)(steps =>
                      sampling.copy(sampleSteps = steps)
                    )
                  },
                  vaeTilingParams = defaults.vaeTilingParams
                )
              )
            ),
            target = (area.width, area.height),
            rows = rows,
            notes = List(
              region.fold(s"source ${source._1}x${source._2}")(selection =>
                s"selection ${selection.width}x${selection.height} at ${selection.x},${selection.y} of ${source._1}x${source._2}, edited through a ${area.width}x${area.height} window at ${area.x},${area.y} and feathered back into the source"
              ) + s", padded to ${reference.getWidth}x${reference.getHeight}; " +
                "each tile the model's image to edit, cut from the picture " +
                "as edited so far and composited over it",
              s"instruction: $instruction"
            ) ++ TiledJobs.loraNote(loras.selections),
            derivation = derivation,
            finish = region.fold[PictureFinish](PictureFinish.AsPainted)(
              PictureFinish.PastedInto(image, area, _)
            ),
            keepTiles = request.keepTiles,
            resumed = resuming.isDefined,
            finishTile = Some(Edit.finishTile)
          )
        }
        upscaler.fold(tiled)(carried)
      }
    }
}

private[postprocess] object Edit {

  /** Above this share of a tile changed, the job log says so: a model that
    * repaints rather than edits lands there — Krea 2 at 75 %, Klein rendering a
    * whole-picture reference at 77 % — but so can a large honest edit, so it is
    * a note, not a failure.
    */
  val RepaintShare: Double = 0.5

  /** A tile the model returned, composited over the crop it was given, with its
    * change mask and the composite kept beside it when tiles are kept.
    */
  val finishTile: TileWindow.FinishTile = (crop, returned) => {
    val result = EditComposite(crop, returned)
    val share = math.round(result.changedShare * 100)
    TileWindow.FinishedTile(
      image = result.image,
      note = Some(
        s"changed $share% of the tile" + Option
          .when(result.shift != (0, 0))(
            s", moved by the model and put back by ${result.shift._1},${result.shift._2} px"
          )
          .getOrElse("") + Option
          .when(result.changedShare > RepaintShare)(
            " — more than half: the model may have repainted rather than edited"
          )
          .getOrElse("")
      ),
      // The composite is the tile itself now; what is kept beside it is what
      // the model returned and the mask drift kept it by.
      kept = List("edit" -> returned, "mask" -> result.mask)
    )
  }
}
