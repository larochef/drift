package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** What drift already has from the sites, looked up by what a browser row shows
  * (François, 2026-09-17: "it would avoid multiple tries to install the same
  * lora", and the same for a checkpoint): the LoRAs of the architecture being
  * browsed, or the models registered anywhere. A search grid that says nothing
  * costs a second install to find out what is already there.
  */
class Installed(
    entries: List[InstalledItem],
    /** What having it is called: a LoRA is installed, a model registered. */
    val verb: String
) {

  /** Everything drift holds, whatever it came from — what a file can join when
    * the site it came from says nothing about it (a file on this machine).
    */
  val all: List[InstalledItem] = entries

  /** The entries holding a file from a matching source, first one first: a
    * repository's files installed apart are several LoRAs, and a new file can
    * join any of them.
    */
  private def matching(matches: ModelSource => Boolean): List[InstalledItem] =
    entries.filter(entry => entry.sources.exists(matches))

  /** Any version of this Civitai model: the grid's tiles are models, and
    * another of its versions being installed is worth knowing before opening
    * it.
    */
  def fromCivitaiModel(modelId: String): List[InstalledItem] = matching {
    case Civitai(id, _, _, _) => id == modelId
    case _                    => false
  }

  /** This Civitai version, which is what installs as one LoRA. */
  def fromCivitaiVersion(versionId: String): List[InstalledItem] = matching {
    case Civitai(_, id, _, _) => id == versionId
    case _                    => false
  }

  /** This Civitai file, which is what registers as one model. */
  def fromCivitaiFile(fileId: String): List[InstalledItem] = matching {
    case Civitai(_, _, id, _) => id == fileId
    case _                    => false
  }

  /** This HuggingFace repository, or one file of it. */
  def fromHuggingFace(
      repo: String,
      filename: Option[String] = None
  ): List[InstalledItem] = matching {
    case HuggingFace(r, f) => r == repo && filename.forall(_ == f)
    case _                 => false
  }

  /** This ModelScope repository, or one file of it. */
  def fromModelScope(
      repo: String,
      filename: Option[String] = None
  ): List[InstalledItem] = matching {
    case ModelScope(r, f) => r == repo && filename.forall(_ == f)
    case _                => false
  }
}

/** One thing drift holds: a LoRA or a model, by the id an install can add files
  * to and the name a chip shows.
  */
case class InstalledItem(id: String, name: String, sources: List[ModelSource])

object Installed {

  /** What a browser opened for neither knows. */
  val none: Installed = new Installed(Nil, "installed")

  /** The LoRAs of one architecture. Only its own count: the same file installed
    * from another architecture's browser is a separate entity with its own
    * files (`specs/09-lora-management.md`).
    */
  def loras(loras: List[Lora]): Installed =
    new Installed(
      loras.map(lora =>
        InstalledItem(lora.id, lora.label, lora.files.map(_.source))
      ),
      "installed"
    )

  /** Every registered model, whatever family it was registered in: the file is
    * the same one, and knowing drift already has it is the point.
    */
  def models(models: List[Model]): Installed =
    new Installed(
      models.map(model =>
        InstalledItem(model.id, model.label, List(model.source))
      ),
      "registered"
    )

  /** The upscalers installed from Civitai (`specs/10`). An upscaler keeps only
    * the model it came from, not the version or the file, so the empty ids
    * below match nothing: a tile is marked, a version or a file row is not.
    */
  def upscalers(upscalers: List[Upscaler]): Installed =
    new Installed(
      upscalers.flatMap(upscaler =>
        upscaler.civitaiModelId.map(modelId =>
          InstalledItem(
            upscaler.id,
            upscaler.label,
            List(Civitai(modelId, "", "", upscaler.fileName))
          )
        )
      ),
      "installed"
    )

  /** The chip a tile or file row carries once drift has it. Reactive: an
    * install made from the open browser marks its tile without a re-search.
    *
    * A modifier, not an element: the chip must be the row's own child, since a
    * span wrapped around it would be the flex item in its place and stand a
    * line box taller than the chips beside it.
    */
  def mark(
      installed: Signal[Installed],
      lookup: Installed => List[InstalledItem],
      mods: Mod[HtmlElement]*
  ): Mod[HtmlElement] =
    child <-- installed.map { known =>
      lookup(known).map(_.name).headOption match {
        case Some(name) =>
          span(
            cls := "tag is-success is-small",
            title := s"Already ${known.verb} as '$name'",
            s"✓ ${known.verb}",
            mods
          )
        case None => emptyNode
      }
    }
}
