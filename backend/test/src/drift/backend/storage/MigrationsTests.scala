package drift.backend.storage

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import com.github.plokhotnyuk.jsoniter_scala.core.{
  readFromStream,
  readFromString
}
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

    test("the migrations drift ships") {
      val shipped =
        Migrations(Files.createTempDirectory("drift-shipped")).defined
      // Every one parses, and the versions are unique and in order.
      assert(shipped.nonEmpty)
      assert(shipped.map(_.version) == shipped.map(_.version).distinct.sorted)
      assert(shipped.forall(_.description.nonEmpty))
      assert(shipped.forall(_.steps.nonEmpty))
    }

    test("2: runners (specs/43)") {
      val dir = Files.createTempDirectory("drift-runners")
      def write(collection: String, id: String, json: String): Unit = {
        Files.createDirectories(dir.resolve(collection))
        Files.writeString(
          dir.resolve(collection).resolve(s"$id.json"),
          json,
          StandardCharsets.UTF_8
        )
      }
      def read(collection: String, id: String): Option[ujson.Value] = {
        val file = dir.resolve(collection).resolve(s"$id.json")
        Option.when(Files.isRegularFile(file))(
          ujson.read(Files.readString(file, StandardCharsets.UTF_8))
        )
      }
      Files.writeString(dir.resolve("schema.json"), """{"version": 1}""")
      write(
        "architectures",
        "mine",
        """{"id": "mine", "tool": "LlamaCpp", "builtIn": false}"""
      )
      write(
        "architectures",
        "krea2",
        """{"id": "krea2", "tool": "SdCpp", "builtIn": true}"""
      )
      write(
        "run-configurations",
        "chat",
        """{"id": "chat", "architectureId": "mine"}"""
      )
      write(
        "run-configurations",
        "image",
        """{"id": "image", "architectureId": "krea2"}"""
      )
      write(
        "run-configurations",
        "orphan",
        """{"id": "orphan", "architectureId": "gone"}"""
      )
      write(
        "runtimes",
        "drift-runner",
        """{"id": "drift-runner", "releaseTag": "drift-runner", "adopted": true}"""
      )
      write(
        "runtimes",
        "mine",
        """{"id": "mine", "releaseTag": "b1", "adopted": true}"""
      )
      Migrations(dir).run()
      // the configurations read their architectures before the built-ins go
      assert(read("run-configurations", "chat").get("runner").str == "LlamaCpp")
      assert(read("run-configurations", "image").get("runner").str == "SdCpp")
      assert(read("run-configurations", "orphan").get("runner").str == "SdCpp")
      assert(
        read("architectures", "mine").get("runners").arr.map(_.str) == Seq(
          "LlamaCpp"
        )
      )
      assert(read("architectures", "krea2").isEmpty) // reseeded at startup
      assert(!read("runtimes", "drift-runner").get("adopted").bool)
      assert(read("runtimes", "mine").get("adopted").bool)
    }

    test("4: Qwen 3.8 27B (specs/42)") {
      val dir = Files.createTempDirectory("drift-qwen38")
      val architectures = Files.createDirectories(dir.resolve("architectures"))
      Files.writeString(dir.resolve("schema.json"), """{"version": 3}""")
      Files.writeString(
        architectures.resolve("qwen3.8-27b.json"),
        """{"id": "qwen3.8-27b", "tool": "LlamaCpp", "builtIn": true}""",
        StandardCharsets.UTF_8
      )
      Files.writeString(
        architectures.resolve("mine.json"),
        """{"id": "mine", "tool": "LlamaCpp", "builtIn": false}""",
        StandardCharsets.UTF_8
      )
      Migrations(dir).run()
      // the stored built-in goes, to be seeded again at startup
      assert(!Files.exists(architectures.resolve("qwen3.8-27b.json")))
      assert(Files.exists(architectures.resolve("mine.json")))
    }

    test("5: FLUX.2 [dev] on the drift runner (specs/42)") {
      val dir = Files.createTempDirectory("drift-flux2-dev")
      val architectures = Files.createDirectories(dir.resolve("architectures"))
      Files.writeString(dir.resolve("schema.json"), """{"version": 4}""")
      Files.writeString(
        architectures.resolve("flux.2-dev.json"),
        """{"id": "flux.2-dev", "tool": "SdCpp", "builtIn": true, "runners": ["SdCpp"]}""",
        StandardCharsets.UTF_8
      )
      Migrations(dir).run()
      // the stored built-in goes, to be seeded again with DriftRunner
      assert(!Files.exists(architectures.resolve("flux.2-dev.json")))
    }

    test("6: Qwen 3.6's MTP head from another file (specs/42)") {
      val dir = Files.createTempDirectory("drift-qwen36-mtp")
      val architectures = Files.createDirectories(dir.resolve("architectures"))
      Files.writeString(dir.resolve("schema.json"), """{"version": 5}""")
      Files.writeString(
        architectures.resolve("qwen3.6-35b-a3b.json"),
        """{"id": "qwen3.6-35b-a3b", "tool": "LlamaCpp", "builtIn": true}""",
        StandardCharsets.UTF_8
      )
      Migrations(dir).run()
      // the stored built-in goes, to be seeded again with its mtp slot
      assert(!Files.exists(architectures.resolve("qwen3.6-35b-a3b.json")))
    }

    test("11: starters on the drift runner where it is installed (specs/46)") {
      val dir = Files.createTempDirectory("drift-starters-on-runner")
      Files.writeString(dir.resolve("schema.json"), """{"version": 10}""")
      def write(collection: String, id: String, json: String): Unit =
        Files.writeString(
          Files
            .createDirectories(dir.resolve(collection))
            .resolve(s"$id.json"),
          json,
          StandardCharsets.UTF_8
        )
      def runnerOf(id: String) =
        ujson
          .read(Files.readString(dir.resolve(s"run-configurations/$id.json")))
          .obj
          .get("runner")
      write(
        "settings",
        "seeded-run-configurations",
        """{"ids": ["starter-pid", "starter-chat"]}"""
      )
      write(
        "architectures",
        "pid",
        """{"id": "pid", "tool": "SdCpp", "runners": ["SdCpp", "DriftRunner"]}"""
      )
      write(
        "architectures",
        "chat",
        """{"id": "chat", "tool": "LlamaCpp", "runners": ["LlamaCpp", "DriftRunner"]}"""
      )
      write(
        "runtimes",
        "drift-runner-images",
        """{"id": "drift-runner-images", "tool": "SdCpp", "valid": true}"""
      )
      write(
        "run-configurations",
        "starter-pid",
        """{"id": "starter-pid", "architectureId": "pid", "runner": "SdCpp"}"""
      )
      // the chat runner is not installed
      write(
        "run-configurations",
        "starter-chat",
        """{"id": "starter-chat", "architectureId": "chat", "runner": "LlamaCpp"}"""
      )
      // the user's own, not a starter
      write(
        "run-configurations",
        "mine",
        """{"id": "mine", "architectureId": "pid", "runner": "SdCpp"}"""
      )
      Migrations(dir).run()
      assert(runnerOf("starter-pid").contains(ujson.Str("DriftRunner")))
      assert(runnerOf("starter-chat").contains(ujson.Str("LlamaCpp")))
      assert(runnerOf("mine").contains(ujson.Str("SdCpp")))
    }

    test("9: a LoRA's sampling settings (specs/49)") {
      val dir = Files.createTempDirectory("drift-lora-sampling")
      val loras = Files.createDirectories(dir.resolve("loras"))
      Files.writeString(dir.resolve("schema.json"), """{"version": 8}""")
      Files.writeString(
        loras.resolve("turbo.json"),
        """{"id": "turbo", "architectureId": "qwen-image-2.1", "label": "Turbo", "files": []}""",
        StandardCharsets.UTF_8
      )
      Migrations(dir).run()
      // an installed LoRA gains the empty settings, and decodes
      val migrated = readFromString[Lora](
        Files.readString(loras.resolve("turbo.json"))
      )
      assert(migrated.sampling == LoraSampling())
    }

    test("the LoRAs drift offers carry their sampling (specs/49)") {
      val stream = getClass.getResourceAsStream("/reference/loras.json")
      val catalog =
        try readFromStream[List[Lora]](stream)
        finally stream.close()
      val viggle = catalog.find(_.id == "qwen-image-2.1-viggle-turbo-6step").get
      assert(
        viggle.sampling == LoraSampling(
          steps = Some(6),
          cfg = Some(1.0),
          flowShift = Some(3.0)
        )
      )
      assert(catalog.forall(_.sampling.nonEmpty))
    }

    test(
      "a LoRA's sampling over a session's, the later LoRA over the earlier"
    ) {
      val base = SampleParameters(
        sampleSteps = 25,
        guidance = GuidanceParameters(txtCfg = 6.0)
      )
      val turbo = LoraSampling(
        steps = Some(6),
        cfg = Some(1.0),
        flowShift = Some(3.0)
      )
      val merged = turbo.over(base)
      assert(merged.sampleSteps == 6)
      assert(merged.guidance.txtCfg == 1.0)
      assert(merged.flowShift == Some(3.0))
      // an empty layer changes nothing
      assert(LoraSampling().over(base) == base)
      // the later LoRA wins the fields it sets, and keeps the others
      val later = LoraSampling(steps = Some(8)).over(turbo)
      assert(later.steps == Some(8) && later.cfg == Some(1.0))
      // the high-noise expert takes its own steps and CFG, and the shift
      val pair = LoraSampling(
        highNoiseSteps = Some(4),
        highNoiseCfg = Some(1.0),
        flowShift = Some(5.0)
      )
      val expert = pair.overHighNoise(base)
      assert(expert.sampleSteps == 4 && expert.guidance.txtCfg == 1.0)
      assert(expert.flowShift == Some(5.0))
    }
  }
}
