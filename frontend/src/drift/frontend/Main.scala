package drift.frontend

import drift.frontend.services.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom

@main def main: Unit = {
  // One socket for every status mirror; AppShell mounts its effects.
  val statusSocketService = StatusSocketService()
  val modelService = ModelService()
  val architectureService = ArchitectureService()
  val runConfigurationService =
    RunConfigurationService(modelService, architectureService)
  val hfService = HuggingFaceService()
  val cacheService = CacheService()
  val downloadService = DownloadService(statusSocketService)
  val runtimeService = RuntimeService(statusSocketService)
  val prerequisites = LaunchPrerequisites(
    runConfigurationService,
    cacheService,
    downloadService,
    runtimeService
  )
  val authTokenService = AuthTokenService()
  val promptTemplateService = PromptTemplateService()
  val sessionService = SessionService(statusSocketService)
  val generationStore = GenerationStore(statusSocketService)
  val generationService =
    GenerationService(statusSocketService, generationStore)
  val assistantService = AssistantService(promptTemplateService)
  val projectService = ProjectService(statusSocketService, generationStore)
  val historyService = HistoryService(statusSocketService, generationStore)
  val postProcessService = PostProcessService(statusSocketService)
  val loraService = LoraService(statusSocketService)
  val upscalerService = UpscalerService(statusSocketService)
  val conversionService = ConversionService(statusSocketService)
  val globalDownloadsService = GlobalDownloadsService(statusSocketService)
  val machineService = MachineService(statusSocketService)
  val locationsService = LocationsService()
  // One instance per service: only one route is mounted at a time, so the
  // browsers and the search page never contend for the same result state.
  val browsers = BrowserServices(
    hfService,
    ModelScopeService(),
    CivitaiService(),
    FileService(),
    authTokenService
  )

  renderOnDomContentLoaded(
    dom.document.getElementById("app"),
    AppShell(
      modelService,
      runConfigurationService,
      architectureService,
      cacheService,
      downloadService,
      prerequisites,
      runtimeService,
      authTokenService,
      promptTemplateService,
      sessionService,
      generationStore,
      generationService,
      assistantService,
      projectService,
      historyService,
      postProcessService,
      loraService,
      upscalerService,
      conversionService,
      globalDownloadsService,
      machineService,
      locationsService,
      LogService(),
      statusSocketService,
      browsers
    ).element
  )
}
