package drift.backend.lora

import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.Path

/** Which entity the files of an install land in (`specs/33-lora-sources.md`):
  * one LoRA per file, or one holding them all, as the request's grouping asks;
  * the id that names it and its folder, free of the ones already taken; and
  * what the site knew about it, kept where the user's own edits will not be
  * overwritten by it.
  *
  * Where the weights come from is `LoraInstalls`; this is only where they are
  * written down.
  */
final private[lora] class LoraPlacement(
    storage: StorageService,
    downloads: LoraDownloads,
    lorasRoot: Path
) {

  /** Saves what an install fetched where its grouping says
    * (`specs/33-lora-sources.md`), and queues every file that is not on disk
    * yet. The response carries the first entity, which is the only one when the
    * files travel together.
    */
  def place(
      architectureId: String,
      grouping: LoraGrouping,
      groupLabel: String,
      sourcePart: String,
      fetched: List[LoraPlacement.Fetched],
      metadata: LoraPlacement.Metadata = LoraPlacement.Metadata(),
      onSaved: Lora => Unit = _ => ()
  ): InstallLoraResponse = grouping match {
    case LoraGrouping.Into(loraId) =>
      storage.get[Lora]("loras", loraId) match {
        case None => LoraManager.failure(s"LoRA '$loraId' no longer exists")
        case Some(target) if target.architectureId != architectureId =>
          LoraManager.failure(
            s"'${target.label}' is a LoRA of another architecture"
          )
        case Some(target) =>
          val lora = withFiles(metadata.refresh(target), fetched.map(_.file))
          save(lora, onSaved)
          InstallLoraResponse(lora = Some(lora))
      }
    case LoraGrouping.Separate =>
      val saved = fetched.map { one =>
        val lora = entityFor(
          architectureId,
          one.ownLabel,
          sourcePart,
          List(one.file),
          metadata,
          ownId = true
        )
        save(lora, onSaved)
        lora
      }
      InstallLoraResponse(lora = saved.headOption)
    case LoraGrouping.Together =>
      val lora = entityFor(
        architectureId,
        groupLabel,
        sourcePart,
        fetched.map(_.file),
        metadata,
        ownId = false
      )
      save(lora, onSaved)
      InstallLoraResponse(lora = Some(lora))
  }

  /** The entity a group of files lands in: the one its id names when it is
    * already there — a re-install keeps the user's tuning and fetches only what
    * is missing — or a new one. `ownId` is for a separate install, which must
    * not join a LoRA that merely shares its name.
    */
  private def entityFor(
      architectureId: String,
      label: String,
      sourcePart: String,
      files: List[LoraFile],
      metadata: LoraPlacement.Metadata,
      ownId: Boolean
  ): Lora = {
    val base = LoraManager.idOf(architectureId, label, sourcePart)
    val id = if (ownId) freeId(base, files) else base
    storage.get[Lora]("loras", id) match {
      case Some(previous) => withFiles(metadata.refresh(previous), files)
      case None           =>
        metadata.fresh(
          Lora(
            id = id,
            architectureId = architectureId,
            label = label,
            files = files,
            createdAt = System.currentTimeMillis()
          )
        )
    }
  }

  /** The first id free for these files: the base one, then `-2`, `-3`… An
    * entity that already holds them all is not in the way — that is the same
    * install again, and it keeps its tuning.
    */
  private def freeId(base: String, files: List[LoraFile]): String = {
    val wanted = files.map(_.fileName).toSet
    LazyList
      .from(1)
      .map(n => if (n == 1) base else s"$base-$n")
      .find(id =>
        storage
          .get[Lora]("loras", id)
          .forall(lora => wanted.subsetOf(lora.files.map(_.fileName).toSet))
      )
      .getOrElse(base)
  }

  private def withFiles(lora: Lora, files: List[LoraFile]): Lora = {
    val known = lora.files.map(_.fileName).toSet
    lora.copy(files =
      lora.files ++ files.filterNot(file => known.contains(file.fileName))
    )
  }

  private def save(lora: Lora, onSaved: Lora => Unit): Unit = {
    storage.save("loras", lora.id, lora)
    onSaved(lora)
    lora.files.foreach(file => downloads.queueDownload(lora, file))
  }
}

object LoraPlacement {

  /** A file an install fetched, and the name its own LoRA would take. */
  case class Fetched(file: LoraFile, ownLabel: String)

  /** What the site knows about a LoRA, which is not what the user tuned: one
    * shape for a LoRA this install creates, another for one it joins.
    */
  case class Metadata(
      fresh: Lora => Lora = identity,
      refresh: Lora => Lora = identity
  )
}
