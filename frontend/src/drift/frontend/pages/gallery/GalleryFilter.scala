package drift.frontend.pages.gallery

import drift.shared.*

/** Which generations a set of filters keeps — the gallery's rules, shared with
  * the picker that offers its entries as inputs
  * (`specs/50-inputs-from-the-gallery.md`). An empty filter keeps everything.
  */
object GalleryFilter {

  def matches(
      configuration: String,
      /** "vid_gen" for videos, anything else non-empty for images. */
      kind: String,
      search: String,
      project: String,
      nsfwShown: Boolean,
      /** The projects flagged NSFW. A generation carries its project, and so
        * does anything derived from it, so the flag reaches upscales and
        * redraws too.
        */
      nsfwProjects: Set[String]
  ): Generation => Boolean = {
    val needle = search.trim.toLowerCase
    generation =>
      (configuration.isEmpty ||
        generation.runConfigurationId == configuration) &&
        (project.isEmpty || generation.projectId.contains(project)) &&
        (nsfwShown || !generation.projectId.exists(nsfwProjects)) &&
        // Derived entries (upscale, resize) are images too.
        (kind.isEmpty ||
          (if (kind == "vid_gen") generation.kind == "vid_gen"
           else generation.kind != "vid_gen")) &&
        (needle.isEmpty ||
          RecordedParameters
            .promptOf(generation)
            .toLowerCase
            .contains(needle))
  }
}
