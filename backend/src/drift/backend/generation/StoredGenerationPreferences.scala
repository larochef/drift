package drift.backend.generation

import drift.backend.storage.{Document, StorageService}
import drift.shared.GenerationPreferences

/** How generations are watched, as the page last chose
  * (`settings/generation-preferences.json`); `GenerationPreferences.Initial`
  * until it has.
  */
final class StoredGenerationPreferences(storage: StorageService) {

  def current: GenerationPreferences =
    storage.read(StoredGenerationPreferences.Stored)

  def save(next: GenerationPreferences): GenerationPreferences =
    storage.write(StoredGenerationPreferences.Stored, next)
}

object StoredGenerationPreferences {
  private val Stored =
    Document("generation-preferences", GenerationPreferences.Initial)
}
