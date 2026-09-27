# 38 — Entity migrations

**Status:** done in code, not yet run live
**Depends on:** 00 (the storage layer and its standing constraints)

drift stores every entity as JSON, one file each, decoded by a codec per type.
Change a field and the stored records no longer fit the codec: they are skipped
with a warning, and a built-in is rewritten by seeding while anything the user
made is simply lost. The answer is not a codec that reads both shapes — that is
legacy the code carries for ever — but a migration that runs once and leaves a
single current format behind (François, 2026-09-19).

## What it does

- The configuration directory carries one schema version, in
  `<config>/schema.json`: `{"version": 1, "migratedAt": "…"}`. A directory
  without the file is version 0.
- At start, before anything is decoded and before the built-ins are seeded,
  every migration above that version runs in order. The version is written
  **last**, so an interrupted run is retried rather than half-recorded.
- Before the first one runs, the directory is copied to
  `<config>/backups/<timestamp>/`. A step that throws stops the whole run: the
  version stays where it was, and the log names the backup.
- A **fresh** install — no entity files at all — is stamped with the current
  version instead of replaying history over empty directories.
- A directory written by a **newer** drift logs a loud warning naming what that
  costs (fields dropped on the next save) and the app continues. Refusing to
  start is the better answer and the intended one; it waits until drift is past
  pre-release.

## Shape

- `backend/.../storage/Migrations.scala`: `Migrations(configDir).run()`, called
  first thing in `StorageService.init`.
- `backend/resources/migrations/index.json` lists the files in order; each is
  `{version, description, steps: [...]}`. A step names a `collection`, an `op`,
  and optionally a `where` — every named field equal to the value given — so a
  migration can touch only the records that need it.
- The operations, which are field-level and deliberately few:

  | op | what it does |
  |---|---|
  | `delete` | drops `field` where present |
  | `add` | writes `value` into `field` where it is missing |
  | `rename` | moves `field` to `to` |
  | `set` | reads `from` (itself by default), looks it up in `map`, writes the result into `field`; a value the map does not name is left alone unless `default` says otherwise |
  | `reseed` | deletes the matching records, for seeding to write again |

- `set` is the one that needs the record to resolve its value — the common
  shape being "this field's old value decides its new one", which is what
  `map` is. Anything irregular is a migration written in Scala against the
  same version sequence, not a bigger dialect: a JSON language grown to cover
  the tail is how these become unmaintainable.
- `reseed` exists because most schema changes touch only user-made records: a
  `builtIn` entity is rewritten from the reference data anyway, so deleting it
  *is* the migration.
- Records are read and written as `ujson` values — the AST the backend gained
  for this. A migration cannot use the entity's codec, since the shape it is
  fixing is the one the codec no longer describes.
- `001-reference-image-use.json` is the first: `Architecture.referenceImages`
  went from a boolean to `ReferenceImageUse` (27), and its `map` turns `true`
  into `Context` and `false` into `Unused`. A record already carrying the enum
  matches nothing in the map and is left alone, so the migration is safe to
  re-run and a no-op on an install that never saw the boolean.
- It is still the only one. `Architecture.sizeMultiple` (01, 27) was added
  without a default on 2026-09-20 and needed none: every architecture on the
  install was built-in, `list` skips a record the codec no longer reads, and
  `seedFromResource` then finds nothing under that id and writes the reference
  version — the case this spec's `reseed` op is named for (François).

## Tested

`backend/test/` is drift's first test module (utest, `./mill backend.test`) and
exists for this: the operations are pure functions over a `ujson.Value`, and
the runner needs nothing but a temporary directory. What it covers:

- Each operation, including the cases that must do **nothing**: `delete` of an
  absent field, `add` over a field already there, `rename` of a field that is
  missing, `set` where the map does not name the value — the last being what
  makes a migration safe to re-run over records already in the new shape.
- `where`, including that a field the record lacks never matches.
- The runner: a fresh directory stamped without a backup, pending migrations
  applied in order, a stored version respected as the starting point (so an
  earlier migration is not replayed), a current directory left alone, a newer
  one left alone, and a failure leaving the version where it was with a backup
  on disk.
- That every migration drift ships parses, with unique versions in order.

The test migrations live in `backend/test/resources/test-migrations/`, so the
runner is exercised on a format the tests own rather than on the ones shipped.

## Notes

- Ordering inside `init` matters: migrate, then seed, then decode. Seeding
  before migrating would rewrite built-ins in the new shape and leave the
  user's records behind in the old one. Seeding itself reads raw JSON for the
  same reason a migration does — see `StorageService.storedOwnership` (01).
- The version is global to the directory, not per collection: entities
  reference each other, and a half-migrated directory is exactly what this
  prevents.
- A step that finds nothing to do logs nothing. Only records actually written
  are counted, so the log says what changed rather than what was considered.
- Backups are never pruned. They are small — JSON entities, no weights — and
  the one time they matter is the time nobody planned for.

## Post-v1

- Refuse to start on a newer schema, once the format stops moving weekly.
- A `drift migrate --dry-run` that prints what each pending migration would
  touch, for a directory worth being careful with.
- Migrations for the outputs' sidecars (`specs/12`), which are entities in
  everything but name and are versioned by nothing today.
