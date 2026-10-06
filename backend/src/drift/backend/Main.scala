package drift.backend

import drift.backend.assistant.{AssistantMedia, AssistantProxy}
import drift.backend.auth.AuthTokens
import drift.backend.cache.{CacheInventory, ModelCache}
import drift.backend.conversion.ConversionManager
import drift.backend.download.*
import drift.backend.lora.LoraManager
import drift.backend.postprocess.PostProcessManager
import drift.backend.projects.ProjectManager
import drift.backend.routes.*
import drift.backend.runtime.{RuntimeCatalog, RuntimeManager}
import drift.backend.sdserver.*
import drift.backend.session.SessionManager
import drift.backend.storage.StorageService
import drift.backend.upscale.UpscalerManager

import scala.concurrent.duration.DurationInt

import com.typesafe.scalalogging.Logger
import ox.*
import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint
import sttp.tapir.server.netty.{NettyConfig, NettySocketConfig}
import sttp.tapir.server.netty.sync.*

// One scope for all of drift: every piece of background work is a fork of it
// (`Background`), so none outlives the server.
@main def main: Unit = supervised {
  val logger = Logger("drift.backend.main")
  val background = Background()
  // Every root drift writes to, resolved once here and handed down: the
  // managers take paths, never defaults (`Locations`).
  val locations = Locations.fromEnvironment()
  logger.info(
    s"""Locations (${locations.appName}):
       |  configuration  ${locations.configDir}
       |  cache          ${locations.cacheRoot}
       |  runtimes       ${locations.runtimesRoot}
       |  outputs        ${locations.outputsRoot}
       |  HuggingFace    ${locations.huggingFaceRoot}""".stripMargin
  )
  val storage = StorageService(locations.configDir)
  storage.init

  val modelCache = ModelCache(
    driftRoot = locations.cacheRoot,
    huggingFaceRoot = locations.huggingFaceRoot
  )

  val cacheInventory = CacheInventory(modelCache, storage)

  val httpClient = java.net.http.HttpClient
    .newBuilder()
    .followRedirects(java.net.http.HttpClient.Redirect.NORMAL)
    .connectTimeout(java.time.Duration.ofSeconds(30))
    .build()
  val downloader = Downloader(httpClient, StallWatches(background))
  val authTokens = AuthTokens(storage)
  val civitaiClient = CivitaiClient(token = () => authTokens.civitai)
  val huggingFaceDownloads = HuggingFaceDownloads(
    huggingFaceRoot = modelCache.huggingFaceRoot,
    driftRoot = modelCache.driftRoot,
    downloader = downloader,
    client = httpClient,
    token = () => authTokens.huggingFace
  )
  val modelScopeApi =
    drift.backend.modelscope.ModelScopeApi(token = () => authTokens.modelScope)
  val modelScopeDownloads = drift.backend.modelscope.ModelScopeDownloads(
    api = modelScopeApi,
    downloader = downloader,
    root = modelCache.modelScopeRoot
  )
  val downloadManager = DownloadManager(
    storage = storage,
    cache = modelCache,
    civitaiClient = civitaiClient,
    huggingFace = huggingFaceDownloads,
    modelScope = modelScopeDownloads,
    downloader = downloader,
    client = httpClient,
    civitaiToken = () => authTokens.civitai,
    background = background
  )

  val runtimeCatalog = RuntimeCatalog(githubToken = sys.env.get("GITHUB_TOKEN"))
  val runtimeManager = RuntimeManager(
    storage,
    downloader,
    runtimeCatalog,
    runtimesRoot = locations.runtimesRoot,
    background = background
  )
  runtimeManager.refreshRunner()
  val loraManager = LoraManager(
    storage = storage,
    civitaiClient = civitaiClient,
    huggingFace = huggingFaceDownloads,
    modelScope = modelScopeDownloads,
    downloader = downloader,
    client = httpClient,
    civitaiToken = () => authTokens.civitai,
    catalog = storage.loraCatalog,
    lorasRoot = locations.lorasRoot,
    background = background
  )
  loraManager.resumeInterrupted()
  val upscalerManager = UpscalerManager(
    storage = storage,
    civitaiClient = civitaiClient,
    downloader = downloader,
    civitaiToken = () => authTokens.civitai,
    upscaleRoot = locations.upscaleRoot,
    background = background
  )
  upscalerManager.resumeInterrupted()
  val sessionManager = SessionManager(
    storage,
    modelCache,
    runtimeManager,
    logsRoot = locations.logsRoot,
    lorasRoot = locations.lorasRoot,
    upscaleRoot = locations.upscaleRoot,
    background = background
  )
  val outputsRoot = locations.outputsRoot
  val projectManager = ProjectManager(storage)
  val generationManager =
    GenerationManager(sessionManager, projectManager, outputsRoot, background)
  val assistantMedia =
    AssistantMedia(
      uploadsRoot = locations.uploadsRoot,
      outputsRoot = outputsRoot
    )
  val assistantProxy = AssistantProxy(
    sessionManager,
    assistantMedia,
    storage,
    locations.lorasRoot
  )
  val generationHistory = GenerationHistory(outputsRoot, generationManager)
  val generationImports = GenerationImports(outputsRoot)
  val projectCovers =
    drift.backend.projects.ProjectCovers(
      storage,
      generationHistory,
      outputsRoot
    )
  // Free play does not survive a restart: whatever the last run left in
  // `outputs/scratch/` was never kept, and the records that pointed at it are
  // gone with the process (`specs/22-free-play-and-scratch-generations.md`).
  generationManager.clearScratch()
  val postProcessManager = PostProcessManager(
    storage = storage,
    outputsRoot = outputsRoot,
    logsRoot = locations.logsRoot,
    history = generationHistory,
    upscalerManager = upscalerManager,
    loraManager = loraManager,
    runtimeManager = runtimeManager,
    sessionManager = sessionManager,
    assistant = assistantProxy,
    background = background
  )

  val conversionManager = ConversionManager(
    storage = storage,
    cache = modelCache,
    runtimeManager = runtimeManager,
    logsRoot = locations.logsRoot,
    background = background
  )

  val apiEndpoints: List[ServerEndpoint[Any, Identity]] =
    architectureEndpoints(storage) ++
      modelEndpoints(storage) ++
      runConfigurationEndpoints(storage) ++
      promptTemplateEndpoints(storage) ++
      cacheEndpoints(storage, modelCache, cacheInventory) ++
      downloadEndpoints(downloadManager) ++
      runtimeEndpoints(storage, runtimeManager, runtimeCatalog) ++
      runtimeRuleEndpoints(storage) ++
      sessionEndpoints(sessionManager, generationManager) ++
      assistantEndpoints(assistantProxy, assistantMedia) ++
      generationEndpoints(
        generationManager,
        OutputPreviews(outputsRoot, locations.cacheRoot)
      ) ++
      scratchEndpoints(generationManager) ++
      historyEndpoints(generationHistory, generationImports) ++
      projectEndpoints(storage, generationHistory, projectCovers) ++
      postProcessEndpoints(postProcessManager) ++
      conversionEndpoints(conversionManager) ++
      loraEndpoints(storage, loraManager) ++
      upscalerEndpoints(upscalerManager) ++
      authEndpoints(storage, authTokens) ++
      fileEndpoints ++
      huggingFaceEndpoints(() => authTokens.huggingFace) ++
      modelScopeEndpoints(modelScopeApi) ++
      civitaiEndpoints(civitaiClient)

  val statusSocket = statusEndpoint(
    sessionManager = sessionManager,
    downloadManager = downloadManager,
    loraManager = loraManager,
    upscalerManager = upscalerManager,
    runtimeManager = runtimeManager,
    generationManager = generationManager,
    postProcessManager = postProcessManager,
    conversionManager = conversionManager
  )

  val askMinutes = AssistantProxy.AskTimeout.toMinutes.toInt
  val nettyConfig =
    NettyConfig.default
      // The assistant's reading of a picture for its redraw is one request
      // that is answered when the reading is in, up to six seconds a tile:
      // minutes on a large picture, where five could cut it off.
      .requestTimeout(askMinutes.minutes + 1.minute)
      // A streamed chat is silent while the model prefills its prompt; the
      // 60 s default dropped long prefills on large models. A reading is
      // silent until it is whole.
      .idleTimeout(askMinutes.minutes + 1.minute)
      .socketConfig(NettySocketConfig.default.withReuseAddress)

  val server = NettySyncServer(nettyConfig)
    .addEndpoints(apiEndpoints)
    .addEndpoint(statusSocket)
    .addEndpoint(assistantChatEndpoint(assistantProxy))
    .addEndpoint(sessionLogEndpoint(sessionManager))
    .addEndpoints(frontendEndpoints)
    .host("0.0.0.0")
    .port(4321)
    .start()

  Runtime.getRuntime.addShutdownHook(new Thread({ () =>
    logger.info("Stopping sessions and server")
    // Children first: killing drift must not orphan an sd-server.
    sessionManager.stopAll()
    server.stop()
  }))

  logger.info(s"Server started on http://localhost:${server.port}")
  never
}
