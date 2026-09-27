package drift.frontend.pages.cache

import drift.frontend.components.*
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** Everything the cache holds, orphans included
  * (`specs/05-model-cache-and-downloads.md`) — "what is using my disk?", and
  * where space is reclaimed: one cache per tab, its files by group, each file
  * assignable to a family (a LoRA folder adoptable by an architecture),
  * unassignable, and deletable.
  */
class OnDiskCacheView(
    /** Which cache is shown, as the page's URL names it. */
    kindTab: Signal[CachedFileKind],
    /** The caches in tab order, each with its label. */
    kindTabs: List[(CachedFileKind, String)],
    /** The URL of a cache's tab, by label. */
    kindUrl: String => String,
    cacheService: CacheService,
    modelService: ModelService,
    architectureService: ArchitectureService,
    runConfigurationService: RunConfigurationService,
    loraService: LoraService,
    conversionService: ConversionService
) extends Component {

  /** The families that already exist — every checkpoint slot's family plus
    * every registered model's — for the assign input's autocomplete.
    */
  private val knownFamilies: Signal[List[String]] =
    architectureService.architectures
      .combineWith(modelService.allModels)
      .map { (architectures, models) =>
        (architectures.flatMap(_.checkpoints.map(_.familyId)) ++
          models.map(_.familyId)).distinct.sorted
      }

  /** model id -> labels of *ready* run configurations assigned to it, so the
    * delete confirmation can name what a removal would break.
    */
  private val readyConfigurationsByModel: Signal[Map[String, List[String]]] =
    Signal
      .combine(
        runConfigurationService.runConfigurations,
        architectureService.architectures,
        modelService.allModels,
        cacheService.statuses
      )
      .map { (configurations, architectures, models, statuses) =>
        val pairs = for {
          configuration <- configurations
          if CommandLine
            .blockers(configuration, architectures, models, statuses)
            .isEmpty
          modelId <- configuration.assignments.values
        } yield modelId -> configuration.label
        pairs.groupMap(_._1)(_._2)
      }

  /** One modal for the whole view; a row's Convert… opens it on that file. */
  private val convertModal = ConvertModelModal(conversionService, knownFamilies)

  /** A weight file sd-cpp can read: safetensors or GGUF, whole, not a LoRA. */
  private def convertible(entry: CachedFileEntry): Boolean = {
    val name = ConvertModelModal.fileName(entry).toLowerCase
    !entry.partial && entry.kind != CachedFileKind.Lora &&
    (name.endsWith(".safetensors") || name.endsWith(".gguf"))
  }

  /** Registers the file as a model in the chosen family. The file is already on
    * disk, so the model is born cached; checkpoint slots wanting that family
    * see it immediately.
    *
    * The model keeps the file's origin — a `HuggingFace` or `Civitai` source,
    * re-downloadable and independent of where the cache puts the file — as the
    * inventory reports it; a file without one is a `Local` path referenced in
    * place.
    */
  private def assignToFamily(
      entry: CachedFileEntry,
      familyId: String,
      models: List[Model]
  ): Unit = {
    // "Sidecar name — file.ext" and "sub/dir/file.ext" both end in the
    // filename, whatever kind of row this is.
    val fileName = entry.label.split(" — ").last.split('/').last
    val (base, extension) = fileName.lastIndexOf('.') match {
      case -1    => (fileName, "")
      case index => (fileName.take(index), fileName.drop(index + 1))
    }
    val slug = base.toLowerCase
      .map(c => if (c.isLetterOrDigit) c else '-')
      .split('-')
      .filter(_.nonEmpty)
      .mkString("-") match {
      case ""    => "model"
      case other => other
    }
    val taken = models.map(_.id).toSet
    val id =
      if (!taken(slug)) slug
      else Iterator.from(2).map(n => s"$slug-$n").filterNot(taken).next()
    val source: ModelSource = entry.source.getOrElse(Local(entry.path))
    modelService.push(
      ModelService.Command.Create(
        Model(
          id = id,
          familyId = familyId,
          label = base,
          source = source,
          format = if (extension.nonEmpty) extension.toLowerCase else "unknown",
          parameters = Map.empty
        )
      )
    )
  }

  private def fileRow(
      entry: CachedFileEntry,
      readyByModel: Map[String, List[String]],
      models: List[Model],
      architectures: List[Architecture]
  ): HtmlElement = {
    val dependingConfigurations =
      entry.referencedBy.flatMap(readyByModel.getOrElse(_, Nil)).distinct
    // A mistaken family assignment is corrected by deleting the registered
    // model(s) again — the file itself stays and goes back to being an
    // orphan, ready for a fresh Assign. Built-in models are not offered; the
    // server refuses to delete them anyway.
    val removableModels = models
      .filter(model => entry.referencedBy.contains(model.id))
      .filterNot(_.builtIn)
    val familyVar = Var("")
    val adoptArchitectureVar = Var("")
    div(
      cls := "level is-mobile mb-1 is-marginless",
      div(
        cls := "level-left",
        div(
          p(cls := "text-primary is-size-7", entry.label),
          p(
            cls := "text-secondary is-size-7",
            if (entry.referencedBy.isEmpty)
              span(cls := "tag is-warning is-small", "orphan")
            else span(s"used by ${entry.referencedBy.mkString(", ")}"),
            if (entry.partial)
              span(cls := "tag is-dark is-small ml-1", "partial download")
            else emptyNode
          )
        )
      ),
      div(
        cls := "level-right",
        // An orphaned LoRA folder is re-attached right here: pick the
        // architecture it belongs to and the entity is rebuilt from the
        // folder's sidecar and files — a `.part` leftover resumes its
        // download.
        if (entry.kind == CachedFileKind.Lora && entry.referencedBy.isEmpty)
          span(
            cls := "mr-2",
            select(
              cls := "select is-small",
              option(value := "", "architecture…"),
              architectures.map(architecture =>
                option(value := architecture.id, architecture.label)
              ),
              onChange.mapToValue --> adoptArchitectureVar
            ),
            button(
              cls := "button is-info is-small ml-1",
              "Adopt",
              disabled <-- adoptArchitectureVar.signal.map(_.isEmpty),
              onClick --> { _ =>
                val architectureId = adoptArchitectureVar.now()
                if (architectureId.nonEmpty)
                  loraService.push(
                    LoraService.Command.Adopt(entry.path, architectureId)
                  )
              }
            )
          )
        else emptyNode,
        // An unclaimed file can be assigned to a family right here — the
        // on-disk view is where orphans surface, so it is also where they
        // are put to work. LoRA files are not slot-fillers, so they get the
        // architecture adopt control above instead of a family input.
        if (
          entry.referencedBy.isEmpty && !entry.partial &&
          entry.kind != CachedFileKind.Lora
        )
          span(
            cls := "mr-2",
            input(
              cls := "input is-small",
              styleAttr := "width: 11em; display: inline-block;",
              placeholder := "family",
              listId := "known-families",
              controlled(
                value <-- familyVar.signal,
                onInput.mapToValue --> familyVar
              )
            ),
            button(
              cls := "button is-info is-small ml-1",
              "Assign",
              disabled <-- familyVar.signal.map(_.trim.isEmpty),
              onClick --> { _ =>
                val family = familyVar.now().trim
                if (family.nonEmpty) assignToFamily(entry, family, models)
              }
            )
          )
        else emptyNode,
        // A GGUF quant of the file, into the same family
        // (`specs/25-model-conversion.md`).
        if (convertible(entry))
          button(
            cls := "button is-small mr-2",
            "Convert…",
            onClick --> (_ => convertModal.show(entry, models))
          )
        else emptyNode,
        if (removableModels.nonEmpty)
          button(
            cls := "button is-warning is-small mr-2",
            "Unassign",
            onClick --> { _ =>
              val warning =
                if (dependingConfigurations.nonEmpty)
                  s"Ready run configuration(s) ${dependingConfigurations
                      .mkString(", ")} depend on this file!\n\n"
                else ""
              val described = removableModels
                .map(model => s"'${model.id}' (family ${model.familyId})")
                .mkString(", ")
              if (
                window.confirm(
                  s"${warning}Unassign ${entry.label} by removing registered model $described? The file stays on disk and can be assigned again."
                )
              )
                removableModels.foreach(model =>
                  modelService.push(ModelService.Command.Delete(model.id))
                )
            }
          )
        else emptyNode,
        span(
          cls := "tag is-small mr-2",
          LaunchBlocker.humanBytes(entry.bytes)
        ),
        button(
          cls := "button is-danger is-small",
          "🗑️",
          onClick --> { _ =>
            val warning =
              if (dependingConfigurations.nonEmpty)
                s"Ready run configuration(s) ${dependingConfigurations
                    .mkString(", ")} depend on this file!\n\n"
              else ""
            if (
              window.confirm(
                s"${warning}Delete ${entry.label} (${LaunchBlocker
                    .humanBytes(entry.bytes)})? Weights are expensive to refetch."
              )
            ) cacheService.push(CacheService.Command.DeleteFile(entry.path))
          }
        )
      )
    )
  }

  lazy val element: HtmlElement = div(
    // One shared datalist: every row's family input autocompletes from the
    // families that already exist, while still accepting a brand-new name — a
    // family needs no declaring, it exists by having members.
    dataList(
      idAttr := "known-families",
      children <-- knownFamilies.map(_.map(family => option(value := family)))
    ),
    convertModal.element,
    div(
      cls := "tabs is-small",
      ul(
        children <-- cacheService.files
          .combineWith(kindTab)
          .map { (files, active) =>
            kindTabs.map { (kind, name) =>
              val count = files.count(_.kind == kind)
              li(
                cls("is-active") := kind == active,
                a(s"$name ($count)", href := kindUrl(name))
              )
            }
          }
      )
    ),
    child <-- cacheService.files.combineWith(kindTab).map { (files, kind) =>
      val matching = files.filter(_.kind == kind)
      div(
        p(
          cls := "text-secondary",
          s"${matching.size} file(s), ${LaunchBlocker
              .humanBytes(matching.map(_.bytes).sum)}"
        ),
        kind match {
          case CachedFileKind.Local =>
            p(
              cls := "text-secondary is-size-7",
              "Local models live under ~/.cache/drift/models — files placed there (or downloads whose sidecar is gone) show up here."
            )
          case CachedFileKind.ModelScope =>
            p(
              cls := "text-secondary is-size-7",
              "ModelScope downloads live under ~/.cache/drift/modelscope, " +
                "one folder per repository. Deleting a row removes that file."
            )
          case CachedFileKind.Lora =>
            p(
              cls := "text-secondary is-size-7",
              "LoRAs live under ~/.cache/drift/loras, one folder per LoRA. " +
                "Deleting a row removes the whole folder — a wan 2.2 " +
                "pair's two files travel together. An orphan folder can " +
                "be re-attached to an architecture with Adopt."
            )
          case _ => emptyNode
        }
      )
    },
    children <-- cacheService.files
      .combineWith(
        kindTab,
        readyConfigurationsByModel,
        modelService.allModels,
        architectureService.architectures
      )
      .map { (files, kind, readyByModel, models, architectures) =>
        files
          .filter(_.kind == kind)
          .groupBy(_.group)
          .toList
          .sortBy(_._1)
          .map { (group, entries) =>
            div(
              cls := "mb-4",
              h2(
                cls := "is-size-6 has-text-weight-bold text-primary mb-1",
                group,
                span(
                  cls := "tag is-light ml-2",
                  LaunchBlocker.humanBytes(entries.map(_.bytes).sum)
                )
              ),
              entries
                .sortBy(_.label)
                .map(fileRow(_, readyByModel, models, architectures))
            )
          }
      }
  )
}
