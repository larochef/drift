package drift.frontend.services

/** The services the model-source browsers need, bundled so they can be threaded
  * from `main` down through `ArchitecturesPage` -> `ArchitectureCard` ->
  * `ModelForm` -> `SourceTypeSelector` as one parameter instead of three.
  *
  * Each browser mounts its own `service.effects`, so opening and closing a
  * modal subscribes and unsubscribes with it. One instance of each service is
  * shared app-wide, which is safe because only one route -- and at most one
  * browser modal -- is mounted at a time.
  */
case class BrowserServices(
    huggingFace: HuggingFaceService,
    modelScope: ModelScopeService,
    civitai: CivitaiService,
    files: FileService
)
