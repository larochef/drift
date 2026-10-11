package drift.backend.storage

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import com.github.plokhotnyuk.jsoniter_scala.core.readFromStream
import utest.*

import drift.shared.*

/** The entity migrations (`specs/38-entity-migrations.md`).
  *
  * Worth testing because the cost of getting one wrong is somebody's stored
  * work, and because it is testable: the operations are pure functions over a
  * JSON value, and the runner needs nothing but a directory.
  */
object MigrationsTests extends TestSuite {

  private def record(json: String): ujson.Value = ujson.read(json)

  private def step(
      op: String,
      field: String = "",
      to: Option[String] = None,
      value: Option[ujson.Value] = None,
      from: Option[String] = None,
      map: Map[String, ujson.Value] = Map.empty,
      default: Option[ujson.Value] = None,
      where: Map[String, ujson.Value] = Map.empty,
      name: Option[String] = None
  ): MigrationStep =
    MigrationStep(
      collection = "widgets",
      op = op,
      field = Option(field).filter(_.nonEmpty),
      to = to,
      value = value,
      from = from,
      name = name,
      map = map,
      default = default,
      where = where
    )

  /** A configuration directory holding `records`, and the migrations under
    * `test-migrations`.
    */
  private def withDirectory[A](
      records: Map[String, String],
      version: Option[Int] = None
  )(check: (Path, Migrations) => A): A = {
    val dir = Files.createTempDirectory("drift-migrations")
    try {
      Files.createDirectories(dir.resolve("widgets"))
      records.foreach((name, json) =>
        Files.writeString(
          dir.resolve("widgets").resolve(s"$name.json"),
          json,
          StandardCharsets.UTF_8
        )
      )
      version.foreach(v =>
        Files.writeString(
          dir.resolve("schema.json"),
          s"""{"version": $v}""",
          StandardCharsets.UTF_8
        )
      )
      check(dir, Migrations(dir, "test-migrations"))
    } finally {
      Files
        .walk(dir)
        .sorted(java.util.Comparator.reverseOrder[Path]())
        .forEach(Files.deleteIfExists(_))
    }
  }

  private def widget(dir: Path, name: String): ujson.Value =
    ujson.read(
      Files.readString(
        dir.resolve("widgets").resolve(s"$name.json"),
        StandardCharsets.UTF_8
      )
    )

  val tests = Tests {

    test("delete") {
      test("drops the field") {
        val out = Migrations.applyTo(
          record("""{"a": 1, "b": 2}"""),
          step("delete", "a")
        )
        assert(out.map(ujson.write(_)).contains("""{"b":2}"""))
      }
      test("leaves a record without it alone") {
        assert(
          Migrations
            .applyTo(record("""{"b": 2}"""), step("delete", "a"))
            .isEmpty
        )
      }
    }

    test("add") {
      test("writes the value where the field is missing") {
        val out = Migrations.applyTo(
          record("""{"a": 1}"""),
          step("add", "b", value = Some(ujson.Num(3)))
        )
        assert(out.map(_("b").num).contains(3.0))
      }
      test("never overwrites a field that is there") {
        assert(
          Migrations
            .applyTo(
              record("""{"b": 9}"""),
              step("add", "b", value = Some(ujson.Num(3)))
            )
            .isEmpty
        )
      }
    }

    test("rename") {
      test("moves the value") {
        val out = Migrations.applyTo(
          record("""{"colour": "red", "size": 2}"""),
          step("rename", "colour", to = Some("color"))
        )
        assert(out.map(_("color").str).contains("red"))
        assert(out.exists(!_.obj.contains("colour")))
      }
      test("does nothing when the field is absent") {
        assert(
          Migrations
            .applyTo(
              record("""{"size": 2}"""),
              step("rename", "colour", to = Some("color"))
            )
            .isEmpty
        )
      }
    }

    test("set") {
      val mapping =
        Map[String, ujson.Value]("true" -> "Context", "false" -> "Unused")

      test("maps the old value to the new one") {
        val out = Migrations.applyTo(
          record("""{"referenceImages": true}"""),
          step(
            "set",
            "referenceImages",
            from = Some("referenceImages"),
            map = mapping
          )
        )
        assert(out.map(_("referenceImages").str).contains("Context"))
      }
      test("leaves a value the map does not name") {
        // The record already migrated: running again must not touch it.
        assert(
          Migrations
            .applyTo(
              record("""{"referenceImages": "Edit"}"""),
              step(
                "set",
                "referenceImages",
                from = Some("referenceImages"),
                map = mapping
              )
            )
            .isEmpty
        )
      }
      test("falls back to the default where one is given") {
        val out = Migrations.applyTo(
          record("""{"quality": "odd"}"""),
          step(
            "set",
            "strength",
            from = Some("quality"),
            map = Map("gentle" -> ujson.Num(0.3)),
            default = Some(ujson.Num(0.4))
          )
        )
        assert(out.map(_("strength").num).contains(0.4))
      }
      test("reads from another field") {
        val out = Migrations.applyTo(
          record("""{"quality": "heavy"}"""),
          step(
            "set",
            "strength",
            from = Some("quality"),
            map = Map("heavy" -> ujson.Num(0.6))
          )
        )
        assert(out.map(_("strength").num).contains(0.6))
      }
    }

    test("coded steps") {
      test("an unknown name changes nothing") {
        assert(
          Migrations
            .applyTo(record("""{"a": 1}"""), step("coded", name = Some("nope")))
            .isEmpty
        )
      }
    }

    test("an unknown operation changes nothing") {
      assert(
        Migrations
          .applyTo(record("""{"a": 1}"""), step("frobnicate", "a"))
          .isEmpty
      )
    }

    test("where") {
      val mine = record("""{"builtIn": false}""")
      val seeded = record("""{"builtIn": true}""")
      assert(Migrations.matches(mine, Map("builtIn" -> ujson.False)))
      assert(!Migrations.matches(seeded, Map("builtIn" -> ujson.False)))
      // No condition means every record.
      assert(Migrations.matches(seeded, Map.empty))
      // A field the record does not carry never matches.
      assert(!Migrations.matches(mine, Map("missing" -> ujson.True)))
    }

    test("the runner") {
      test("stamps a fresh directory without touching anything") {
        withDirectory(Map.empty) { (dir, migrations) =>
          assert(migrations.currentVersion == 0)
          assert(migrations.run() == 2)
          assert(migrations.currentVersion == 2)
          // Nothing was migrated, so nothing was backed up.
          assert(!Files.isDirectory(dir.resolve("backups")))
        }
      }

      test("applies every pending migration in order") {
        withDirectory(
          Map(
            "one" -> """{"id": "one", "colour": "red", "mode": true, "builtIn": false}""",
            "two" -> """{"id": "two", "colour": "blue", "mode": true, "builtIn": true}"""
          )
        ) { (dir, migrations) =>
          assert(migrations.run() == 2)
          val one = widget(dir, "one")
          // 001 renamed and added, 002 mapped — and only for the user's record.
          assert(one("color").str == "red")
          assert(!one.obj.contains("colour"))
          assert(one("size").num == 3)
          assert(one("mode").str == "Loud")
          assert(widget(dir, "two")("mode").bool)
          assert(Files.isDirectory(dir.resolve("backups")))
        }
      }

      test("starts from the stored version, not from zero") {
        withDirectory(
          Map(
            "one" -> """{"id": "one", "colour": "red", "mode": true, "builtIn": false}"""
          ),
          version = Some(1)
        ) { (dir, migrations) =>
          assert(migrations.run() == 2)
          val one = widget(dir, "one")
          // 002 ran; 001 did not, so the old field is untouched.
          assert(one("mode").str == "Loud")
          assert(one("colour").str == "red")
          assert(!one.obj.contains("size"))
        }
      }

      test("does nothing when the directory is already current") {
        withDirectory(
          Map("one" -> """{"id": "one", "colour": "red"}"""),
          version = Some(2)
        ) { (dir, migrations) =>
          assert(migrations.run() == 2)
          assert(widget(dir, "one")("colour").str == "red")
          assert(!Files.isDirectory(dir.resolve("backups")))
        }
      }

      test("leaves a newer directory alone") {
        withDirectory(
          Map("one" -> """{"id": "one", "colour": "red"}"""),
          version = Some(99)
        ) { (dir, migrations) =>
          // Warns loudly and carries on, for now (`specs/38`).
          assert(migrations.run() == 99)
          assert(migrations.currentVersion == 99)
          assert(widget(dir, "one")("colour").str == "red")
        }
      }

      test("a failure keeps the old version and leaves a backup") {
        withDirectory(
          Map(
            "one" -> """{"id": "one", "colour": "red", "mode": true, "builtIn": false}""",
            "broken" -> """{"id": "broken", not json at all"""
          )
        ) { (dir, migrations) =>
          val failed =
            try { migrations.run(); false }
            catch { case _: Throwable => true }
          assert(failed)
          assert(migrations.currentVersion == 0)
          assert(Files.isDirectory(dir.resolve("backups")))
        }
      }
    }

    test("no migration is shipped: the stored records are the schema") {
      val shipped =
        Migrations(Files.createTempDirectory("drift-shipped")).defined
      assert(shipped.isEmpty)
    }

    test("the LoRAs drift offers carry their sampling (specs/49)") {
      val stream = getClass.getResourceAsStream("/reference/loras.json")
      val catalog =
        try readFromStream[List[Lora]](stream)
        finally stream.close()
      val viggle = catalog.find(_.id == "qwen-image-2.1-viggle-turbo-6step").get
      assert(
        viggle.sampling == LoraSampling.none.copy(
          steps = Some(6),
          cfg = Some(1.0),
          sigmas = List(1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
        )
      )
      assert(catalog.forall(_.sampling.nonEmpty))
    }
  }
}
