package drift.backend.generation

import drift.backend.storage.StorageService
import drift.shared.GenerationPreferences

/** How generations are watched, as the page last chose
  * (`settings/generation-preferences.json`); `GenerationPreferences.Initial`
  * until it has.
  */
final class StoredGenerationPreferences(storage: StorageService) {

  def current: GenerationPreferences =
    storage
      .get[GenerationPreferences]("settings", "generation-preferences")
      .getOrElse(GenerationPreferences.Initial)

  def save(next: GenerationPreferences): GenerationPreferences =
    storage.save("settings", "generation-preferences", next)
}
