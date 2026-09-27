package drift.backend.modelscope

import drift.backend.routes.ModelDescriptions
import drift.shared.*

import java.net.URI
import java.net.http.*
import java.time.Duration
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.*

// ModelScope's answers, reduced to what drift reads. Its fields are
// capitalized; everything is optional, since its models are shaped by many
// hands (a base model list can be `null`, `[""]` or carry `@revision`s).
private case class Cover(
    url: Option[String] = None,
    prompt: Option[String] = None
)
private case class MuseVersion(coverImages: List[Cover] = Nil)
private case class Muse(versions: List[MuseVersion] = Nil)
private case class RawModel(
    @named("Path") path: String = "",
    @named("Name") name: String = "",
    @named("Downloads") downloads: Option[Long] = None,
    @named("Stars") stars: Option[Long] = None,
    @named("AigcType") aigcType: Option[String] = None,
    @named("BaseModel") baseModel: Option[List[String]] = None,
    @named("LastUpdatedTime") lastUpdatedTime: Option[Long] = None,
    @named("MuseInfo") muse: Muse = Muse(),
    @named("ReadMeContent") readMe: Option[String] = None
)
private case class RawModels(
    @named("Models") models: List[RawModel] = Nil,
    @named("TotalCount") totalCount: Option[Int] = None
)
private case class SearchData(@named("Model") model: Option[RawModels] = None)
private case class SearchAnswer(
    @named("Code") code: Option[Long] = None,
    @named("Message") message: Option[String] = None,
    @named("Data") data: Option[SearchData] = None
)
private case class InfoAnswer(
    @named("Code") code: Option[Long] = None,
    @named("Message") message: Option[String] = None,
    @named("Data") data: Option[RawModel] = None
)
private case class RawFile(
    @named("Path") path: String = "",
    @named("Type") kind: Option[String] = None,
    @named("Size") size: Option[Long] = None,
    @named("Sha256") sha256: Option[String] = None
)
private case class FilesData(@named("Files") files: List[RawFile] = Nil)
private case class FilesAnswer(
    @named("Code") code: Option[Long] = None,
    @named("Message") message: Option[String] = None,
    @named("Data") data: Option[FilesData] = None
)

private case class SearchCriterion(
    category: String,
    predicate: String,
    values: List[String],
    sub_values: List[String]
)
private case class SearchBody(
    @named("PageSize") pageSize: Int,
    @named("PageNumber") pageNumber: Int,
    @named("SortBy") sortBy: String,
    @named("Target") target: String,
    @named("Name") name: String,
    @named("SingleCriterion") singleCriterion: List[SearchCriterion],
    @named("Criterion") criteria: List[SearchCriterion]
)

private val museCodec: JsonValueCodec[Muse] = JsonCodecMaker.make

/** A model's `MuseInfo` is an object in a search and a JSON string in a model's
  * own info; anything else counts as no Muse information.
  */
private given lenientMuseCodec: JsonValueCodec[Muse] =
  new JsonValueCodec[Muse] {
    val nullValue: Muse = Muse()

    def encodeValue(x: Muse, out: JsonWriter): Unit =
      museCodec.encodeValue(x, out)

    def decodeValue(in: JsonReader, default: Muse): Muse =
      if (in.isNextToken('"')) {
        in.rollbackToken()
        val text = in.readString(null)
        try readFromString[Muse](text)(using museCodec)
        catch { case NonFatal(_) => nullValue }
      } else {
        in.rollbackToken()
        if (in.isNextToken('{')) {
          in.rollbackToken()
          museCodec.decodeValue(in, nullValue)
        } else {
          in.rollbackToken()
          in.skip()
          nullValue
        }
      }
  }

private given searchAnswerCodec: JsonValueCodec[SearchAnswer] =
  JsonCodecMaker.make
private given infoAnswerCodec: JsonValueCodec[InfoAnswer] = JsonCodecMaker.make
private given filesAnswerCodec: JsonValueCodec[FilesAnswer] =
  JsonCodecMaker.make
// Every field written: the search expects its empty criteria lists.
private given searchBodyCodec: JsonValueCodec[SearchBody] = JsonCodecMaker.make(
  CodecMakerConfig.withTransientEmpty(false).withTransientDefault(false)
)

/** drift's client of ModelScope's API (`specs/37-modelscope.md`): the search, a
  * model's info and files, its README. ModelScope copies HuggingFace's shapes
  * in places but answers in its own ways, so it has its own client, never
  * HuggingFace's.
  */
final class ModelScopeApi(
    /** The active ModelScope token, resolved per request so a token saved in
      * Settings applies at once. A wrong one does not stop public requests.
      */
    token: () => Option[String] = () => None
) {

  /** Never follows a redirect: the API answers in place, and a download's
    * redirect is resolved by hand — ModelScope's CDN location can carry raw
    * spaces, which the JDK refuses to follow, and the token must not travel
    * with it.
    */
  private val client = HttpClient
    .newBuilder()
    .followRedirects(HttpClient.Redirect.NEVER)
    .connectTimeout(Duration.ofSeconds(20))
    .build()

  /** The token as ModelScope's SDK sends it: the `Authorization` header the
    * newer endpoints read and the session cookies the older ones do, both to
    * modelscope.cn only.
    */
  private def credentials: Map[String, String] =
    token().map(_.trim).filter(_.nonEmpty).fold(Map.empty[String, String]) {
      value =>
        Map(
          "Authorization" -> s"Bearer $value",
          "Cookie" -> s"m_session_id=$value; modelscope_session=$value"
        )
    }

  private def authenticated(builder: HttpRequest.Builder): HttpRequest.Builder =
    credentials.foldLeft(builder)((request, header) =>
      request.header(header._1, header._2)
    )

  private val Host = "https://modelscope.cn"
  private val PageSize = 24

  /** One page of a search, `page` counting from 1. */
  def search(
      query: String,
      page: Int,
      sort: Option[String],
      baseModels: List[String],
      lorasOnly: Boolean
  ): Either[String, ModelScopeSearchPage] = {
    def criterion(category: String, values: List[String]) =
      SearchCriterion(category, "contains", values, Nil)
    val body = SearchBody(
      pageSize = PageSize,
      pageNumber = page.max(1),
      sortBy = sort
        .filter(key => ModelScopeSorts.exists(_._1 == key))
        .getOrElse("DownloadsCount"),
      target = "",
      name = query.trim,
      singleCriterion = Nil,
      criteria = Option
        .when(baseModels.nonEmpty)(criterion("base_model", baseModels))
        .toList ++
        Option.when(lorasOnly)(criterion("aigc_type", List("LoRA"))).toList
    )
    send(
      HttpRequest
        .newBuilder(URI.create(s"$Host/api/v1/dolphin/models"))
        .header("Content-Type", "application/json")
        .PUT(HttpRequest.BodyPublishers.ofString(writeToString(body)))
    ).flatMap(json => decode[SearchAnswer](json)).flatMap { answer =>
      answer.code match {
        case Some(200) =>
          val found = answer.data.flatMap(_.model)
          Right(
            ModelScopeSearchPage(
              models = found.toList.flatMap(_.models).map(infoOf),
              total = found.flatMap(_.totalCount).getOrElse(0)
            )
          )
        case _ => Left(failureOf(answer.code, answer.message))
      }
    }
  }

  /** A repository's files and examples; `None` when it does not exist. */
  def detail(repo: String): Either[String, Option[ModelScopeModelDetail]] =
    model(repo).flatMap {
      case None      => Right(None)
      case Some(raw) =>
        files(repo).map(found =>
          Some(
            ModelScopeModelDetail(
              id = repo,
              files = found,
              examples = examplesOf(raw, found)
            )
          )
        )
    }

  /** The README, rendered like a HuggingFace card, its `<Gallery />` made of
    * the versions' cover images.
    */
  def card(repo: String): Either[String, ModelScopeModelCard] =
    model(repo).map {
      case None      => ModelScopeModelCard()
      case Some(raw) =>
        ModelScopeModelCard(html =
          raw.readMe
            .filter(_.trim.nonEmpty)
            .map(markdown =>
              ModelDescriptions.fromMarkdown(
                ModelDescriptions.withGallery(markdown, coversOf(raw)),
                s"$Host/models/$repo/resolve/master/"
              )
            )
            .filter(_.trim.nonEmpty)
        )
    }

  /** Every file of the repository, with its size and sha256. */
  def files(repo: String): Either[String, List[ModelScopeFileInfo]] =
    send(
      HttpRequest.newBuilder(
        URI.create(
          s"$Host/api/v1/models/$repo/repo/files?Recursive=true&Revision=master"
        )
      )
    ).flatMap(json => decode[FilesAnswer](json)).flatMap { answer =>
      answer.code match {
        case Some(200) =>
          Right(
            answer.data.toList
              .flatMap(_.files)
              .filter(_.kind.forall(_ == "blob"))
              .map(file =>
                ModelScopeFileInfo(
                  path = file.path,
                  size = file.size,
                  sha256 = file.sha256.map(_.toLowerCase)
                )
              )
          )
        case _ => Left(failureOf(answer.code, answer.message))
      }
    }

  /** Where a file is actually served from, and the headers to fetch it with.
    * ModelScope redirects a download to its CDN with a signed, short-lived
    * query, sometimes naming the file with raw spaces: the location is read
    * here and made a valid URL, fetched without credentials — the signature is
    * the authorization. A file served in place is fetched with them. A download
    * resolves this again each time it starts, so a resume never reuses an
    * expired signature.
    */
  def downloadRequest(
      repo: String,
      path: String
  ): Either[String, (String, Map[String, String])] = {
    val url = fileUrl(repo, path)
    try {
      val response = client.send(
        authenticated(HttpRequest.newBuilder(URI.create(url)))
          .timeout(Duration.ofSeconds(30))
          .header("User-Agent", "drift/0.1.0")
          // A HEAD is answered 200 without the redirect; a one-byte GET gets
          // it, and costs a byte when the file is served in place.
          .header("Range", "bytes=0-0")
          .GET()
          .build(),
        HttpResponse.BodyHandlers.discarding()
      )
      if (response.statusCode / 100 == 3)
        response
          .headers()
          .firstValue("Location")
          .map[Either[String, (String, Map[String, String])]](location =>
            Right(ModelScopeApi.validUrl(location) -> Map.empty)
          )
          .orElse(Left(s"ModelScope redirected '$path' nowhere"))
      else if (response.statusCode == 200 || response.statusCode == 206)
        Right(url -> credentials)
      else if (response.statusCode == 401 || response.statusCode == 403)
        Left(
          s"ModelScope refused '$path' (HTTP ${response.statusCode}): the " +
            "repository is private or gated — set a ModelScope token in Settings"
        )
      else
        Left(s"ModelScope answered HTTP ${response.statusCode} for '$path'")
    } catch {
      case NonFatal(err) =>
        Left(
          s"ModelScope could not be reached: ${Option(err.getMessage).getOrElse(err.toString)}"
        )
    }
  }

  /** A file's URL on modelscope.cn, its path encoded segment by segment. */
  def fileUrl(repo: String, path: String): String =
    URI(
      "https",
      "modelscope.cn",
      s"/models/$repo/resolve/master/$path",
      null
    ).toASCIIString

  private def model(repo: String): Either[String, Option[RawModel]] =
    send(HttpRequest.newBuilder(URI.create(s"$Host/api/v1/models/$repo")))
      .flatMap(json => decode[InfoAnswer](json))
      .flatMap { answer =>
        answer.code match {
          case Some(200) => Right(answer.data)
          case Some(404) => Right(None)
          case _         => Left(failureOf(answer.code, answer.message))
        }
      }

  private def infoOf(raw: RawModel): ModelScopeModelInfo = {
    val examples = examplesOf(raw, Nil)
    ModelScopeModelInfo(
      id = s"${raw.path}/${raw.name}",
      downloads = raw.downloads.getOrElse(0L),
      stars = raw.stars.getOrElse(0L),
      lora = raw.aigcType.exists(_.equalsIgnoreCase("LoRA")),
      baseModels = raw.baseModel.toList.flatten.filter(_.nonEmpty),
      updatedAt = raw.lastUpdatedTime,
      previews = examples.take(4),
      exampleCount = examples.size
    )
  }

  private def coversOf(raw: RawModel): List[ModelExample] =
    raw.muse.versions.flatMap(_.coverImages).flatMap { cover =>
      cover.url
        .map(_.trim)
        .filter(_.nonEmpty)
        .map(url =>
          ModelExample(
            url = url,
            video = isVideo(url.takeWhile(_ != '?')),
            prompt = cover.prompt.map(_.trim).filter(_.nonEmpty)
          )
        )
    }

  /** The versions' cover images, with their prompts; the repository's media
    * files only when there are none — a repository mirrored onto ModelScope has
    * its sample files copied as covers, and they would show twice.
    */
  private def examplesOf(
      raw: RawModel,
      files: List[ModelScopeFileInfo]
  ): List[ModelExample] = {
    val repo = s"${raw.path}/${raw.name}"
    val covers = coversOf(raw)
    val found =
      if (covers.nonEmpty) covers
      else
        files
          .map(_.path)
          .filter(isMedia)
          .map(path => ModelExample(fileUrl(repo, path), isVideo(path)))
    found.distinctBy(_.url)
  }

  private val ImageExtensions = List(".png", ".jpg", ".jpeg", ".webp", ".gif")
  private val VideoExtensions = List(".mp4", ".webm", ".mov")

  private def isVideo(path: String): Boolean =
    VideoExtensions.exists(path.toLowerCase.endsWith)

  private def isMedia(path: String): Boolean =
    isVideo(path) || ImageExtensions.exists(path.toLowerCase.endsWith)

  private def send(builder: HttpRequest.Builder): Either[String, String] =
    try {
      val response = client.send(
        authenticated(builder)
          .timeout(Duration.ofSeconds(30))
          .header("Accept", "application/json")
          .header("User-Agent", "drift/0.1.0")
          .build(),
        HttpResponse.BodyHandlers.ofString()
      )
      // ModelScope answers its own errors in the body, often with a 200.
      if (response.statusCode >= 500)
        Left(s"ModelScope answered HTTP ${response.statusCode}")
      else Right(response.body)
    } catch {
      case NonFatal(err) =>
        Left(
          s"ModelScope could not be reached: ${Option(err.getMessage).getOrElse(err.toString)}"
        )
    }

  private def decode[A](json: String)(using
      JsonValueCodec[A]
  ): Either[String, A] =
    try Right(readFromString[A](json))
    catch {
      case NonFatal(err) =>
        Left(s"ModelScope's answer could not be read: ${err.getMessage}")
    }

  private def failureOf(code: Option[Long], message: Option[String]): String =
    s"ModelScope answered ${code.fold("without a code")(c => s"code $c")}" +
      message.filter(_.nonEmpty).fold("")(m => s": $m")
}

object ModelScopeApi {

  /** A URL as a server wrote it, with what a URI may not hold — spaces,
    * controls, non-ASCII, `"<>\^`{|}` — percent-encoded; `%` sequences already
    * there are kept.
    */
  def validUrl(raw: String): String =
    raw.flatMap { character =>
      if (
        character <= ' ' || character >= 127 || "\"<>\\^`{|}".contains(
          character
        )
      )
        character.toString
          .getBytes(java.nio.charset.StandardCharsets.UTF_8)
          .map(byte => f"%%${byte & 0xff}%02X")
          .mkString
      else character.toString
    }
}
