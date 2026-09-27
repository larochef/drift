package drift.backend.storage

import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.time.Instant
import scala.io.Source
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** One thing a migration does to one collection
  * (`specs/38-entity-migrations.md`).
  *
  * The four operations are the ones a field-level change needs — and no more:
  * anything irregular is a `coded` step (`Migrations.coded`) rather than a
  * dialect grown until it is a language. `Reseed` is the fifth because most
  * schema changes touch only user-made records: a built-in is rewritten from
  * the reference data anyway, so deleting it is the whole migration.
  */
private[storage] case class MigrationStep(
    collection: String,
    op: String,
    /** The field acted on, for every op but `reseed`. */
    field: Option[String] = None,
    /** `rename`: where it goes. */
    to: Option[String] = None,
    /** `add`: the value written when the field is missing. */
    value: Option[ujson.Value] = None,
    /** `set`: the field the new value is read from — itself, usually. */
    from: Option[String] = None,
    /** `coded`: which function in `Migrations.coded` rewrites the record. */
    name: Option[String] = None,
    /** `set`: old value (as text) to new value. A value the map does not name
      * is left alone unless `default` says otherwise, so a record already
      * migrated is not touched twice.
      */
    map: Map[String, ujson.Value] = Map.empty,
    default: Option[ujson.Value] = None,
    /** Only records whose fields all match are touched. */
    where: Map[String, ujson.Value] = Map.empty
)

private[storage] case class Migration(
    version: Int,
    description: String,
    steps: List[MigrationStep] = Nil
)

/** Versioned migrations over the stored entities, the way flyway versions a
  * database (`specs/38-entity-migrations.md`, François, 2026-09-19).
  *
  * The configuration directory carries one version in `schema.json`; each
  * migration above it runs in order, and the version is written last. Before
  * any of them runs the directory is copied to `backups/<timestamp>`, and a
  * step that throws stops the whole run — a half-migrated directory is the
  * thing this exists to prevent, and the backup is what the log points at.
  *
  * Migrations replace tolerant codecs: a format changes once, the records are
  * rewritten once, and the code keeps exactly one shape in mind.
  */
final class Migrations(configDir: Path, resource: String = "migrations") {
  private val logger = Logger[Migrations]

  private def versionFile: Path = configDir.resolve("schema.json")

  /** The version on disk; a directory without the file is version 0. */
  def currentVersion: Int =
    try
      if (!Files.isRegularFile(versionFile)) 0
      else
        ujson
          .read(Files.readString(versionFile, StandardCharsets.UTF_8))
          .obj
          .get("version")
          .map(_.num.toInt)
          .getOrElse(0)
    catch {
      case NonFatal(err) =>
        logger.warn(s"Cannot read ${versionFile.getFileName}", err)
        0
    }

  private def writeVersion(version: Int): Unit =
    Files.writeString(
      versionFile,
      ujson.write(
        ujson.Obj(
          "version" -> version,
          "migratedAt" -> Instant.now().toString
        ),
        indent = 2
      ) + "\n",
      StandardCharsets.UTF_8
    )

  /** Whether anything has been stored yet: a fresh install is stamped with the
    * current version rather than replaying history over empty directories.
    */
  private def isEmpty: Boolean =
    !Files.isDirectory(configDir) || {
      val entities = Files
        .walk(configDir)
        .iterator()
        .asScala
        .filter(p => p.toString.endsWith(".json"))
        .filterNot(_.getFileName.toString == "schema.json")
      !entities.hasNext
    }

  /** Every migration drift ships, in order. */
  def defined: List[Migration] = Migrations.load(resource)

  /** Brings the configuration directory up to date. Answers the version it is
    * left at.
    */
  def run(): Int = {
    val all = defined
    val target = all.map(_.version).maxOption.getOrElse(0)
    val from = currentVersion

    if (from > target) {
      // A directory written by a newer drift. Refusing to start is the honest
      // answer and will be the answer later; while drift is pre-release the
      // warning is loud and the app goes on (François, 2026-09-19).
      logger.warn(
        s"!!! The configuration in $configDir is at schema version $from, " +
          s"and this build knows only $target. It was written by a newer " +
          "drift. Fields it added are dropped on the next save, and fields " +
          "it renamed may decode as missing. Run the newer build, or start " +
          "from a copy."
      )
      from
    } else if (from == target) from
    else if (isEmpty) {
      logger.info(s"Fresh configuration: stamping schema version $target")
      writeVersion(target)
      target
    } else {
      val pending = all.filter(_.version > from).sortBy(_.version)
      val backup = backUp()
      logger.info(
        s"Migrating $configDir from schema $from to $target: " +
          s"${pending.size} migration(s), backup in $backup"
      )
      try {
        pending.foreach { migration =>
          logger.info(s"  ${migration.version}: ${migration.description}")
          migration.steps.foreach(apply)
        }
        writeVersion(target)
        target
      } catch {
        case NonFatal(err) =>
          logger.error(
            s"!!! Migration failed; the configuration is part-migrated and " +
              s"its version is still $from. A copy of it as it was is in " +
              s"$backup — restore that before running again.",
            err
          )
          throw err
      }
    }
  }

  /** The directory as it stands, copied beside itself. */
  private def backUp(): Path = {
    val target = configDir
      .resolve("backups")
      .resolve(Instant.now().toString.replace(":", "-"))
    Files.createDirectories(target)
    Files
      .list(configDir)
      .iterator()
      .asScala
      .filterNot(_.getFileName.toString == "backups")
      .foreach { entry =>
        val destination = target.resolve(entry.getFileName.toString)
        if (Files.isDirectory(entry)) {
          Files.createDirectories(destination)
          Files
            .list(entry)
            .iterator()
            .asScala
            .foreach(file =>
              Files.copy(file, destination.resolve(file.getFileName.toString))
            )
        } else Files.copy(entry, destination)
      }
    target
  }

  /** One step over every record of its collection. */
  private def apply(step: MigrationStep): Unit = {
    val dir = configDir.resolve(step.collection)
    if (!Files.isDirectory(dir)) return
    val files = Files
      .list(dir)
      .iterator()
      .asScala
      .filter(_.toString.endsWith(".json"))
      .toList
    var touched = 0
    files.foreach { file =>
      val text = Files.readString(file, StandardCharsets.UTF_8)
      val record = ujson.read(text)
      if (Migrations.matches(record, step.where)) {
        if (step.op == "reseed") {
          Files.delete(file)
          touched += 1
        } else
          (if (step.op == "coded") coded(record, step)
           else Migrations.applyTo(record, step)).foreach { updated =>
            Files.writeString(
              file,
              ujson.write(updated, indent = 2) + "\n",
              StandardCharsets.UTF_8
            )
            touched += 1
          }
      }
    }
    if (touched > 0)
      logger.info(
        s"    ${step.op} on ${step.collection}: $touched record(s)"
      )
  }

  /** A `coded` step: the function it names, which may read other records. */
  private def coded(
      record: ujson.Value,
      step: MigrationStep
  ): Option[ujson.Value] =
    step.name.flatMap(Migrations.coded.get) match {
      case Some(rewrite) => rewrite(record, stored)
      case None          =>
        logger.warn(
          s"Migration names an unknown coded step '${step.name.getOrElse("")}', skipped"
        )
        None
    }

  /** A record of another collection, as it stands at this step. */
  private def stored(collection: String, id: String): Option[ujson.Value] = {
    val file = configDir.resolve(collection).resolve(s"$id.json")
    Option.when(Files.isRegularFile(file))(
      ujson.read(Files.readString(file, StandardCharsets.UTF_8))
    )
  }
}

private[storage] object Migrations {
  private val logger = Logger("drift.backend.storage.Migrations")

  /** Whether a record matches a step's `where`: every named field equal to the
    * value given. No field named means every record.
    */
  def matches(record: ujson.Value, where: Map[String, ujson.Value]): Boolean =
    where.forall((field, expected) => record.obj.get(field).contains(expected))

  /** The record after `step`, or `None` when the step leaves it alone — so a
    * record already in the new shape is not rewritten.
    */
  def applyTo(
      record: ujson.Value,
      step: MigrationStep
  ): Option[ujson.Value] = {
    val fields = record.obj
    val field = step.field.getOrElse("")
    step.op match {
      case "delete" =>
        Option.when(fields.contains(field)) {
          fields.remove(field)
          record
        }
      case "add" =>
        Option.when(!fields.contains(field) && step.value.isDefined) {
          fields(field) = step.value.get
          record
        }
      case "rename" =>
        val to = step.to.getOrElse("")
        Option.when(fields.contains(field) && to.nonEmpty) {
          fields(to) = fields.remove(field).get
          record
        }
      case "set" =>
        val source = step.from.getOrElse(field)
        val current = fields.get(source).map(asKey)
        val replacement = current.flatMap(step.map.get).orElse(step.default)
        replacement.flatMap(value =>
          Option.unless(fields.get(field).contains(value)) {
            fields(field) = value
            record
          }
        )
      case "coded" =>
        logger.warn("A coded step runs over the directory, not a lone record")
        None
      case other =>
        logger.warn(s"Unknown migration operation '$other', skipped")
        None
    }
  }

  /** What JSON cannot say. A step `{"op": "coded", "name": "…"}` runs the
    * function registered here: the version, the order and the collection stay
    * in the migration file, and only the rewriting is Scala. Answering `None`
    * leaves the record alone.
    *
    * Empty, and meant to stay nearly so — the four field operations cover
    * field-level change, and this is for the shape that splits one record into
    * two, or reads a file beside it, or anything else a dialect would have to
    * grow a language to express.
    */
  val coded: Map[
    String,
    (ujson.Value, (String, String) => Option[ujson.Value]) => Option[
      ujson.Value
    ]
  ] = Map(
    // `specs/43`: a configuration runs on its architecture's upstream engine,
    // whose name is its tool's ("SdCpp", "LlamaCpp"); sd-cpp when the
    // architecture is gone, as a launch reads it
    "runner-from-architecture" -> { (record, stored) =>
      Option.when(!record.obj.contains("runner")) {
        val tool = record.obj
          .get("architectureId")
          .flatMap(id => stored("architectures", id.str))
          .flatMap(_.obj.get("tool"))
          .getOrElse(ujson.Str("SdCpp"))
        record.obj("runner") = tool
        record
      }
    }
  )

  /** A stored value as the text a `map` is keyed by: `true`, `7`, `"Context"`
    * all read as themselves, without their JSON quotes.
    */
  private def asKey(value: ujson.Value): String = value match {
    case ujson.Str(text) => text
    case other           => ujson.write(other)
  }

  /** The migrations shipped in `resources/<resource>/`, ordered. */
  def load(resource: String): List[Migration] =
    Option(getClass.getClassLoader.getResourceAsStream(s"$resource/index.json"))
      .map { stream =>
        val names =
          try ujson.read(Source.fromInputStream(stream, "UTF-8").mkString).arr
          finally stream.close()
        names.toList.flatMap(name => read(s"$resource/${name.str}"))
      }
      .getOrElse(Nil)
      .sortBy(_.version)

  private def read(path: String): Option[Migration] =
    Option(getClass.getClassLoader.getResourceAsStream(path)).flatMap {
      stream =>
        try {
          val json =
            ujson.read(Source.fromInputStream(stream, "UTF-8").mkString)
          Some(
            Migration(
              version = json("version").num.toInt,
              description = json("description").str,
              steps = json.obj
                .get("steps")
                .map(_.arr.toList.map(step))
                .getOrElse(Nil)
            )
          )
        } catch {
          case NonFatal(err) =>
            logger.error(s"Cannot read migration $path", err)
            None
        } finally stream.close()
    }

  private def step(json: ujson.Value): MigrationStep = {
    def str(name: String) = json.obj.get(name).map(_.str)
    MigrationStep(
      collection = json("collection").str,
      op = json("op").str,
      field = str("field"),
      to = str("to"),
      name = str("name"),
      value = json.obj.get("value"),
      from = str("from"),
      map = json.obj
        .get("map")
        .map(_.obj.toMap)
        .getOrElse(Map.empty),
      default = json.obj.get("default"),
      where = json.obj
        .get("where")
        .map(_.obj.toMap)
        .getOrElse(Map.empty)
    )
  }
}
