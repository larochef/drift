package drift.backend.storage

import drift.shared.*

import java.nio.charset.StandardCharsets
import java.nio.file.*
import scala.io.Source
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.typesafe.scalalogging.Logger

/** A stored entity as seeding needs it: who owns it, and the bytes it holds
  * (`StorageService.storedRecords`).
  */
private case class StoredRecord(builtIn: Boolean, text: String)

class StorageService(basePath: Path) {
  private val logger = Logger[StorageService]
  private val entityTypes =
    Seq(
      "architectures",
      "models",
      "run-configurations",
      "runtimes",
      "settings",
      "auth-tokens",
      // A project's conversation with the assistant (`specs/20`), kept apart
      // from `projects/`, every file of which is read as a `Project`.
      "conversations"
    )
  entityTypes.foreach(t => Files.createDirectories(basePath.resolve(t)))

  private def read[T: JsonValueCodec](p: Path): Option[T] =
    try Some(readFromString[T](Files.readString(p, StandardCharsets.UTF_8)))
    catch {
      case NonFatal(err) =>
        logger.warn(s"Failed to decode ${p.getFileName}: ${err.getMessage}")
        None
    }

  def list[T](entityType: String)(using JsonValueCodec[T]): List[T] = {
    val dir = basePath.resolve(entityType)
    if (!Files.isDirectory(dir)) Nil
    else
      Files
        .list(dir)
        .iterator()
        .asScala
        .filter(_.toString.endsWith(".json"))
        .flatMap(read[T])
        .toList
  }

  def get[T](entityType: String, id: String)(using
      JsonValueCodec[T]
  ): Option[T] = {
    val p = basePath.resolve(entityType).resolve(s"$id.json")
    if (Files.exists(p)) read[T](p) else None
  }

  /** True when the entity has a file on disk, even if that file no longer
    * decodes into `T`. Callers that only need to know whether an id exists must
    * use this rather than `get(...).isDefined`, which cannot tell a missing
    * entity from an unreadable one.
    */
  def exists(entityType: String, id: String): Boolean =
    Files.exists(basePath.resolve(entityType).resolve(s"$id.json"))

  /** The exact bytes an entity is stored as. Seeding compares what is on disk
    * with this before writing, so the two must be the one function.
    */
  private def serialize[T](value: T)(using JsonValueCodec[T]): String =
    writeToString(value, WriterConfig.withIndentionStep(2))

  def save[T](entityType: String, id: String, value: T)(using
      JsonValueCodec[T]
  ): T = {
    val file = basePath.resolve(entityType).resolve(s"$id.json")
    // Ids may contain '/' (e.g. legacy "owner/repo" model ids); create every
    // parent directory so Files.writeString does not hit NoSuchFileException.
    Files.createDirectories(file.getParent)
    // Write to a sibling temp file and move it into place: an interrupted write
    // must never leave a half-written file, because `list` skips whatever it
    // cannot decode and the entity would silently vanish from the UI.
    val tmp =
      Files.createTempFile(file.getParent, s".${file.getFileName}", ".tmp")
    try {
      Files.writeString(tmp, serialize(value), StandardCharsets.UTF_8)
      try
        Files.move(
          tmp,
          file,
          StandardCopyOption.REPLACE_EXISTING,
          StandardCopyOption.ATOMIC_MOVE
        )
      catch {
        case _: AtomicMoveNotSupportedException =>
          Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING)
      }
    } catch {
      case e: Throwable =>
        Files.deleteIfExists(tmp)
        throw e
    }
    value
  }

  def delete(entityType: String, id: String): Boolean =
    Files.deleteIfExists(basePath.resolve(entityType).resolve(s"$id.json"))

  /** Every stored record of a type: its `id` to what seeding needs to know —
    * whether it is the user's, and the bytes it holds.
    *
    * Read as raw JSON, never through `T`'s codec. The records on disk were
    * written by the previous build, and a field added to the entity since makes
    * every one of them fail to decode — one warning per file, about files the
    * next lines rewrite from the reference anyway (François, 2026-09-20, adding
    * `sizeMultiple`). It is the same rule migrations follow for the same reason
    * (`specs/38-entity-migrations.md`), and `id` and `builtIn` outlive any
    * codec. A file that is not JSON at all still warns: that one is broken
    * rather than merely old.
    */
  private def storedRecords(
      entityType: String
  ): Map[String, StoredRecord] = {
    val dir = basePath.resolve(entityType)
    if (!Files.isDirectory(dir)) Map.empty
    else {
      val files = Files.list(dir)
      try
        files
          .iterator()
          .asScala
          .filter(_.toString.endsWith(".json"))
          .flatMap { file =>
            try {
              val text = Files.readString(file, StandardCharsets.UTF_8)
              val record = ujson.read(text)
              val builtIn =
                record.obj.get("builtIn").flatMap(_.boolOpt).getOrElse(false)
              record.obj
                .get("id")
                .flatMap(_.strOpt)
                .map(id => id -> StoredRecord(builtIn, text))
            } catch {
              case NonFatal(err) =>
                logger.warn(
                  s"Failed to read ${file.getFileName}: ${err.getMessage}"
                )
                None
            }
          }
          .toMap
      finally files.close()
    }
  }

  /** Seeds one reference file into its entity directory.
    *
    * A record is written only when it is not already exactly what would be
    * written: the stored bytes are compared with the bytes the reference item
    * serialises to, which is why `save` and this share one `serialize`. A start
    * where nothing changed writes nothing, and no marker file is needed to know
    * that — the files themselves are the record of what was seeded (François,
    * 2026-09-20). Comparing against the bytes rather than against a note of
    * what we once wrote also repairs a file that was corrupted or edited by
    * hand.
    */
  private def seedFromResource[T](
      entityType: String,
      resourcePath: String,
      extractId: T => String
  )(using JsonValueCodec[T], JsonValueCodec[List[T]]): Unit = {
    val refItems: List[T] = loadResource(resourcePath)
    // A resource that is missing, empty or undecodable must change nothing:
    // the retirement step below would otherwise read "the reference lists no
    // built-ins" and delete every one of them.
    if (refItems.isEmpty) {
      logger.warn(s"$resourcePath read as empty: nothing seeded or retired")
      return
    }
    val stored: Map[String, StoredRecord] = storedRecords(entityType)
    val refIds = refItems.map(extractId).toSet

    // Built-in entities are the reference file's, not the user's: the UI
    // refuses to edit or delete them, so a changed reference (a fixed flag on
    // a built-in architecture) must reach an existing config too. Only an
    // entity the user created under a reference id is left alone.
    val written = refItems.count { item =>
      val id = extractId(item)
      stored.get(id) match {
        case Some(record) if !record.builtIn                => false
        case Some(record) if record.text == serialize(item) => false
        case _ => save(entityType, id, item); true
      }
    }
    if (written > 0)
      logger.info(s"seeded $written built-in $entityType from $resourcePath")

    // A built-in that left the reference goes with it: it was never the
    // user's, and a run configuration still pointing at it fails loudly in
    // the UI rather than keeping a retired architecture alive.
    stored
      .collect {
        case (id, record) if record.builtIn && !refIds.contains(id) => id
      }
      .foreach { id =>
        logger.info(s"removing retired built-in $entityType '$id'")
        delete(entityType, id)
      }
  }

  private def loadResource[T](
      resourcePath: String
  )(using JsonValueCodec[List[T]]): List[T] = {
    Option(
      Thread
        .currentThread()
        .getContextClassLoader
        .getResourceAsStream(resourcePath)
    ).map { stream =>
      try {
        val content = Source.fromInputStream(stream, "UTF-8").mkString
        try {
          val items = readFromString[List[T]](content)
          logger.debug(s"Loaded ${items.size} items from $resourcePath")
          items
        } catch {
          case NonFatal(err) =>
            logger.warn(s"Failed to decode $resourcePath: ${err.getMessage}")
            Nil
        }
      } finally stream.close()
    }.getOrElse(Nil)
  }

  def init: Unit = {
    // Stored records are brought to the shape this build expects before
    // anything decodes them, and before seeding rewrites the built-ins
    // (`specs/38-entity-migrations.md`).
    Migrations(basePath).run()
    seedFromResource[Architecture](
      "architectures",
      "reference/architectures.json",
      _.id
    )
    // One runnable implementation per built-in architecture's checkpoints
    // (`specs/23-seeded-vendor-models.md`), so a fresh install can assign a
    // configuration without hunting for repo paths. Same rule as above: a
    // model the user created under a reference id is theirs and is left
    // alone, and a built-in that leaves the reference file goes with it.
    seedFromResource[Model](
      "models",
      "reference/models.json",
      _.id
    )
    // The prompts drift puts in front of models (`specs/32-prompt-library.md`):
    // built-ins re-seed like architectures, user variants are copies.
    seedFromResource[PromptTemplate](
      "prompt-templates",
      "reference/prompt-templates.json",
      _.id
    )
    tagUntaggedArchitectures()
  }

  /** Fills in `tags` for architectures registered before the field existed
    * (`specs/01-architecture-registry.md`).
    *
    * Built-in architectures do not need this: `seedFromResource` above rewrites
    * them from the reference file, tags included. This is for the ones the user
    * created, which seeding never touches — they would otherwise sit untagged
    * and appear under no filter but "All".
    *
    * Only empty tag lists are touched, so an architecture that has been tagged
    * is never overwritten; the cost of that rule is that emptying an
    * architecture's tags deliberately gets them derived again next start.
    */
  private def tagUntaggedArchitectures(): Unit = {
    val untagged = list[Architecture]("architectures").filter(_.tags.isEmpty)
    untagged.foreach { architecture =>
      val tags = ArchitectureTags.derive(architecture)
      save("architectures", architecture.id, architecture.copy(tags = tags))
    }
    if (untagged.nonEmpty)
      logger.info(
        s"Tagged ${untagged.size} architecture(s) that predate the tags field"
      )
  }

  /** What each build wants and refuses (`specs/16-parameter-resolution.md`).
    * Read straight from the resource, never seeded into the config directory:
    * these describe upstream builds rather than the user's choices, so there is
    * nothing here for them to edit and nothing to migrate when drift ships a
    * new rule.
    */
  def runtimeRules: List[RuntimeRule] =
    loadResource[RuntimeRule]("reference/runtime-rules.json")

  /** The official LoRAs drift offers to install (`specs/33-lora-sources.md`).
    * Read straight from the resource like the runtime rules: an offer is not
    * the user's until installed, and an installed one is an ordinary entity the
    * user tunes and deletes, so nothing is seeded.
    */
  def loraCatalog: List[Lora] =
    loadResource[Lora]("reference/loras.json")
}
