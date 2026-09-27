package drift.backend.routes

import drift.backend.sdserver.GenerationHistory
import drift.backend.storage.StorageService
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

def projectEndpoints(
    storage: StorageService,
    history: GenerationHistory,
    covers: drift.backend.projects.ProjectCovers
): List[ServerEndpoint[Any, Identity]] =
  endpointsFor[Project](
    storage,
    "projects",
    _.id,
    listProjects,
    getProject,
    createProject,
    updateProject,
    deleteProject,
    // The conversation goes with its project: it was scratch about the
    // project, not work to keep (`specs/20`).
    onDeleted = (project: Project) => {
      storage.delete("conversations", project.id)
      ()
    },
    // The versions and `lastUsedAt` are the backend's own record, appended by
    // `ProjectManager` as generations run. A rename, a brief or the NSFW flag
    // travels as a whole `Project`, and the client's copy of the version list
    // is stale the moment a generation appends to it - honouring it would drop
    // the version that was just created and hand its number out again
    // (`specs/19-projects-and-prompt-versions.md`).
    mergeUpdate = (stored: Project, incoming: Project) =>
      incoming.copy(versions = stored.versions, lastUsedAt = stored.lastUsedAt)
  ) ++ List(
    getConversation.serverLogicSuccess[Identity] { projectId =>
      storage
        .get[Conversation]("conversations", projectId)
        .getOrElse(Conversation(projectId))
    },
    saveConversation.serverLogicSuccess[Identity] { (projectId, conversation) =>
      val stored = conversation.copy(projectId = projectId)
      // A project deleted meanwhile keeps no conversation behind.
      if (storage.exists("projects", projectId))
        storage.save[Conversation]("conversations", projectId, stored)
      else stored
    },
    // A full walk of the sidecars: the day index is the only index there is,
    // and today's volumes make that fine (`specs/19-…`).
    listProjectGenerations.serverLogicSuccess[Identity] { projectId =>
      history.days
        .flatMap(day => history.day(day.date))
        .filter(_.projectId.contains(projectId))
        .sortBy(-_.submittedAt)
    },
    getProjectCover.serverLogic[Identity] { projectId =>
      covers.cover(projectId).toRight(())
    },
    // The same walk, deleting instead of listing: the day is known here, so
    // each generation goes through the history's own delete — files, inputs
    // and sidecar — and the answer names what actually went.
    deleteProjectGenerations.serverLogicSuccess[Identity] { projectId =>
      history.days.flatMap { day =>
        history
          .day(day.date)
          .filter(_.projectId.contains(projectId))
          .filter(generation => history.delete(day.date, generation.id))
          .map(_.id)
      }
    }
  )
