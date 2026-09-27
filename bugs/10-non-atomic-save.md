# Bug 10 — Non-atomic saves: a crash mid-write corrupts the entity, and `list` silently drops it

**Status:** fixed
**Severity:** low/medium (corruption is rare, but its effect is an entity vanishing with only a warn log)
**Files:** `backend/src/drift/backend/storage/StorageService.scala`

## Context

All persistence goes through `StorageService`: one JSON file per entity at `~/.config/drift/<entityType>/<id>.json`. `list[T]` reads every `.json` in the directory and decodes it; files that fail to decode are skipped with a `logger.warn` and the entity disappears from the UI until the file is fixed.

## Symptom

If the JVM dies (or writes are interrupted) while `save` is writing a file, the on-disk JSON is truncated/partial. On next read, `list` skips it (warn log "Failed to decode ...") — the entity is gone from the app with no error surfaced anywhere.

## Root cause

```scala
// StorageService.scala (~lines 48–57)
def save[T](entityType: String, id: String, value: T)(using Encoder[T]): T = {
  val dir = basePath.resolve(entityType)
  Files.createDirectories(dir)
  Files.writeString(dir.resolve(s"$id.json"), value.asJson.spaces2, StandardCharsets.UTF_8)
  value
}
```

In-place overwrite, no temp file + rename. Related: the target's parent directory is only ever created for the entity dir itself, so an id containing `/` (see Bug 3) throws `NoSuchFileException` instead of being handled.

## Suggested fix

Write-then-rename in the same directory (crash-safe and preserves the existing file on failure):

```scala
def save[T](entityType: String, id: String, value: T)(using Encoder[T]): T = {
  val dir = basePath.resolve(entityType)
  val target = dir.resolve(s"$id.json")
  Files.createDirectories(target.getParent)
  val tmp = Files.createTempFile(dir, s".${id}-", ".tmp")
  try {
    Files.writeString(tmp, value.asJson.spaces2, StandardCharsets.UTF_8)
    Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
  } finally {
    Files.deleteIfExists(tmp)
  }
  value
}
```

Optionally, surface decode failures to the frontend (e.g. include skipped files in the list response or a health endpoint) so a corrupted entity is visible rather than invisible.

## Verification

- Code review of the write path (atomic move + cleanup of the temp file on failure).
- Optional: start the server, trigger a save, kill -9 the JVM mid-save (loop until it lands mid-write), restart — the previous (complete) file must still be intact.
