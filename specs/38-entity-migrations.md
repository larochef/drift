# 38 — Entity migrations

**Status:** done
**Depends on:** 00 (the storage layer and its standing constraints)

drift stores every entity as JSON, one file each, decoded by a codec per type.
Change a field and the stored records no longer fit the codec: they are skipped
with a warning, and a built-in is rewritten by seeding while anything the user
made is simply lost. The answer is not a codec that reads both shapes — that is
legacy the code carries for ever — but a migration that runs once and leaves a
single current format behind.

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
- **None is shipped today** (2026-10-11): the seventeen written while the
  format moved weekly were retired together, the stored records rewritten in
  full and the version set back to 0. The next change of a stored field is
  migration 1.
- **A stored record says every field** (`storedRecord`, the codec setting of
  every stored type): an empty list and an absent value are written, and a
  record lacking a field does not decode — it is skipped with a warning, as
  loud as a record gets. The entity types carry no default values, so a field
  added to one does not compile until every place that builds a record says
  it, and does not load until a migration adds it to the stored ones. Two
  limits: an optional field may still be left out of a file (the JSON library
  reads it as absent), and the requests a record embeds — a prompt version's
  generation parameters, a paused job's request — keep their defaults, an
  absent field there meaning "whatever runs it decides".
- A record that may not exist yet starts from a named value, not from a
  constructor without arguments: `SessionSettings.initial`,
  `LoraSampling.none`, `AssistantAppends.standard`, `Conversation.empty`,
  `RuntimeSelection.none`, `AuthTokenSelection.none`.

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
- That no migration is shipped.

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
