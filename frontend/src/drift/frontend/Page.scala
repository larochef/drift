package drift.frontend

import drift.shared.ProjectKind

import frontroute.BrowserNavigation

sealed trait Page {
  def path: String
  def label: String
  def icon: String = ""
  def navigate(): Unit = BrowserNavigation.pushState(url = path)
}

object Page {

  /** Models and how they are run (François, 2026-09-10): the run configurations
    * of both runners and the architectures behind them, on one page. Replaces
    * the separate Inference, Assistant and Architectures entries, whose paths
    * still route here.
    */
  case object Models extends Page {
    val path = "/models"
    val label = "Models"
    override val icon = "🎨"
  }

  /** Projects (`specs/19-projects-and-prompt-versions.md`): the list, and one
    * project's workspace — versions, the generation panel and the assistant.
    */
  case object Projects extends Page {
    val path = "/projects"
    val label = "Projects"
    override val icon = "\uD83D\uDCC1"
  }

  /** The Sandbox (`specs/47-sandbox.md`): trying a model, nothing kept. The
    * rest of the path names what is being made — `/sandbox/video`.
    */
  case object Sandbox extends Page {
    val path = "/sandbox"
    val label = "Sandbox"
    override val icon = "\uD83E\uDDEA"

    def navigateTo(kind: ProjectKind): Unit =
      BrowserNavigation.pushState(url = s"$path/${kind.noun}")
  }

  case class ProjectWorkspace(id: String) extends Page {
    val path = s"/projects/$id"
    val label = "Project"
  }

  /** The gallery (`specs/12-gallery.md`): everything drift ever made. Its own
    * sidebar entry since 2026-09-10 — it is somewhere you go, not a detour off
    * the page that happened to make the images.
    */
  case object Gallery extends Page {
    val path = "/gallery"
    val label = "Gallery"
    override val icon = "\uD83D\uDDBC\uFE0F"
  }

  case object ModelCache extends Page {
    val path = "/model-cache"
    val label = "Model Cache"
    override val icon = "💾"
  }

  case object Settings extends Page {
    val path = "/settings"
    val label = "Settings"
    override val icon = "⚙️"
  }

  /** The sidebar, in the order the work happens: projects are what drift is for
    * and are what `/` shows (François, 2026-09-10).
    */
  val navPages: List[Page] = List(
    Projects,
    Sandbox,
    Gallery,
    Models,
    ModelCache,
    Settings
  )
}
