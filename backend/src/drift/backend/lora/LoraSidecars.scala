package drift.backend.lora

import drift.shared.*

import java.net.URI
import java.net.http.*
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.time.Instant
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.{writeToString, WriterConfig}
import com.typesafe.scalalogging.Logger

/** What a LoRA folder carries beside its weights
  * (`specs/09-lora-management.md`, `specs/33-lora-sources.md`): the sidecar
  * that lets the on-disk view show what a folder is without asking Civitai
  * again, and the couple of preview images that go with it.
  *
  * All of it regenerable and all of it best effort: a preview that does not
  * come down must never fail an install.
  */
final private[lora] class LoraSidecars(client: HttpClient) {

  private val logger = Logger[LoraSidecars]

  /** Regenerable decoration for the on-disk view, plus up to two previews —
    * best effort, a failed preview must not fail an install.
    */
  def writeSidecar(
      folder: Path,
      detail: CivitaiModelDetail,
      version: CivitaiModelVersion,
      lora: Lora
  ): Unit =
    try {
      Files.createDirectories(folder)
      val previews = version.images
        .filter(_.`type` == "image")
        .take(2)
        .zipWithIndex
        .flatMap { case (image, index) =>
          fetchPreview(image.url, folder, index + 1)
        }
      val metadata = ModelMetadata(
        source = "civitai-lora",
        modelId = detail.id.toString,
        versionId = version.id.toString,
        fileId = lora.files
          .map(_.source)
          .collect { case Civitai(_, _, fileId, _) => fileId }
          .mkString(","),
        name = detail.name,
        versionName = version.name,
        creator = detail.creator.flatMap(_.username),
        description = detail.description,
        tags = detail.tags,
        baseModel = version.baseModel,
        trainedWords = version.trainedWords,
        publishedAt = version.publishedAt,
        nsfwLevel = detail.nsfwLevel,
        previews = previews,
        fetchedAt = Instant.now().toString
      )
      Files.write(
        folder.resolve("drift-lora.json"),
        writeToString(metadata, WriterConfig.withIndentionStep(2))
          .getBytes(StandardCharsets.UTF_8)
      )
    } catch {
      case NonFatal(err) =>
        logger.warn(s"LoRA sidecar for '${lora.id}' not written", err)
    }

  private def fetchPreview(
      url: String,
      directory: Path,
      index: Int
  ): Option[String] =
    try {
      val small = url.replace("/original=true/", "/width=450/original=true/")
      val extension =
        url.split('.').lastOption.filter(_.length <= 4).getOrElse("jpg")
      val name = s"preview-$index.$extension"
      val request = HttpRequest
        .newBuilder(URI.create(small))
        .timeout(java.time.Duration.ofSeconds(20))
        .GET()
        .build()
      val response =
        client.send(request, HttpResponse.BodyHandlers.ofByteArray())
      if (response.statusCode == 200) {
        Files.write(directory.resolve(name), response.body)
        Some(name)
      } else None
    } catch { case NonFatal(_) => None }
}
