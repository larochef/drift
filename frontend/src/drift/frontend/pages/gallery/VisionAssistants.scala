package drift.frontend.pages.gallery

import drift.frontend.pages.gallery.VisionAssistants.*
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The assistants that can read a picture (`specs/52-auto-redraw.md`), as the
  * gallery sees them: the chat configurations whose model has a vision
  * projector, the chat sessions that are live, and the launch of one — through
  * the same commands every other launch from the gallery goes through.
  */
class VisionAssistants(
    sessionService: SessionService,
    runConfigurationService: RunConfigurationService,
    /** The project a launch from here is made for, recorded on its session. */
    launchingProject: Option[String]
) {

  private val chatConfigurations: Signal[List[(RunConfiguration, Boolean)]] =
    runConfigurationService.runConfigurations
      .combineWith(runConfigurationService.architectures)
      .map { (configurations, architectures) =>
        val byId = architectures
          .filter(_.tool == RuntimeTool.LlamaCpp)
          .map(a => a.id -> a)
          .toMap
        configurations.flatMap(configuration =>
          byId
            .get(configuration.architectureId)
            .map(a => configuration -> a.readsImages(configuration))
        )
      }
      .distinct

  /** The chat configurations that read images, the last one run first. */
  val configurations: Signal[List[RunConfiguration]] =
    chatConfigurations
      .map(_.collect { case (c, true) => c }.sortBy(-_.lastUsedAt))
      .distinct

  /** The live chat sessions, oldest first — the order the backend looks for one
    * that reads images in (`AssistantProxy.visionSession`).
    */
  private val live: Signal[List[Live]] =
    sessionService.sessions
      .combineWith(chatConfigurations)
      .map { (sessions, configurations) =>
        sessions.values
          .filter(s => s.status.isActive && s.tool == RuntimeTool.LlamaCpp)
          .toList
          .sortBy(_.startedAt)
          .map { session =>
            val known =
              configurations.find(_._1.id == session.runConfigurationId)
            Live(
              session.id,
              known.fold(session.runConfigurationId)(_._1.label),
              session.status == SessionStatus.Ready,
              known.exists(_._2),
              session.memoryWarning
            )
          }
      }
      .distinct

  /** Where a reading stands: who would answer it, or what is in the way. */
  val state: Signal[State] =
    live
      .combineWith(configurations)
      .map { (sessions, readers) =>
        val reading = sessions.filter(_.readsImages)
        reading
          .find(_.ready)
          .map(State.Ready(_))
          .orElse(reading.headOption.map(State.Loading(_)))
          .orElse(Option.when(readers.isEmpty)(State.NoneInstalled))
          .orElse(sessions.headOption.map(State.Other(_)))
          .getOrElse(State.NoneRunning)
      }
      .distinct

  /** Starts `configurationId`, in the place of `replacing` when one is named:
    * never beside it — two chat models loaded is memory nobody asked for.
    */
  def start(configurationId: String, replacing: Option[Live]): Unit = {
    replacing.foreach(session =>
      sessionService.push(SessionService.Command.Stop(session.sessionId))
    )
    sessionService.push(
      SessionService.Command.Launch(configurationId, None, launchingProject)
    )
  }
}

object VisionAssistants {

  /** A live chat session: its model's name, whether it serves yet, whether it
    * reads images, and what its launch said about memory.
    */
  case class Live(
      sessionId: String,
      label: String,
      ready: Boolean,
      readsImages: Boolean,
      memoryWarning: Option[String]
  )

  enum State {

    /** One that reads images is serving: a reading is asked of it. */
    case Ready(session: Live)

    /** One that reads images is loading. */
    case Loading(session: Live)

    /** A chat model is live, and it does not read images. */
    case Other(session: Live)

    /** No chat model is live. */
    case NoneRunning

    /** No chat configuration reads images at all. */
    case NoneInstalled
  }
}
