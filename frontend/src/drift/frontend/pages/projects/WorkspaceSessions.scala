package drift.frontend.pages.projects

import drift.frontend.services.{RunConfigurationService, SessionService}
import drift.shared.*

import com.raquo.laminar.api.L.*

/** What the workspace reads about the sessions it runs and the configurations
  * they come from — each narrowed to what its reader shows, so a session's
  * progress ticks stop here instead of rebuilding the page's pieces.
  */
class WorkspaceSessions(
    sessionService: SessionService,
    runConfigurationService: RunConfigurationService
) {

  /** The live session of a tool as the ids that name it: the session and its
    * configuration.
    */
  def liveKey(tool: RuntimeTool): Signal[Option[(String, String)]] =
    sessionService.sessions
      .map(
        _.values
          .filter(s => s.status.isActive && s.tool == tool)
          .toList
          .sortBy(_.startedAt)
          .headOption
          .map(s => (s.id, s.runConfigurationId))
      )
      .distinct

  /** The status of a tool's live session. */
  def liveStatus(tool: RuntimeTool): Signal[Option[SessionStatus]] =
    sessionService.sessions
      .map(
        _.values
          .filter(s => s.status.isActive && s.tool == tool)
          .toList
          .sortBy(_.startedAt)
          .headOption
          .map(_.status)
      )
      .distinct

  /** The run configurations of a tool's architectures, by label, each with
    * whether its architecture makes the project's kind
    * (`specs/31-project-kinds.md`) — always, without a kind.
    */
  def configurationsOf(
      tool: RuntimeTool,
      kind: Signal[Option[ProjectKind]] = Val(None)
  ): Signal[List[(RunConfiguration, Boolean)]] =
    runConfigurationService.runConfigurations
      .combineWith(runConfigurationService.architectures, kind)
      .map { (configurations, architectures, kind) =>
        val byId =
          architectures.filter(_.tool == tool).map(a => a.id -> a).toMap
        configurations
          .flatMap(c =>
            byId
              .get(c.architectureId)
              .map(architecture => c -> kind.forall(_.accepts(architecture)))
          )
          .sortBy(_._1.label)
      }
      .distinct

  def labelOf(configurationId: String): Signal[String] =
    runConfigurationService.runConfigurations
      .map(
        _.find(_.id == configurationId).map(_.label).getOrElse(configurationId)
      )
      .distinct
}
