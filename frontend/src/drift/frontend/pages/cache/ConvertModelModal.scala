package drift.frontend.pages.cache

import drift.frontend.components.{Component, ScrollLock}
import drift.frontend.services.ConversionService
import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

/** "Convert…" on an on-disk file (`specs/25-model-conversion.md`): what the
  * file holds, the target type with the size it should come out at, the family
  * the result joins, the output name — and, folded, sd-cpp's raw per-tensor
  * type rules and a thread count. Submitting queues the job; the backend's
  * refusal, if any, shows here beside the form.
  */
class ConvertModelModal(
    conversionService: ConversionService,
    /** The families that already exist, for the family input's autocomplete
      * (the `known-families` datalist of the on-disk view).
      */
    knownFamilies: Signal[List[String]]
) extends Component {
  import ConvertModelModal.*

  private val entry = Var(Option.empty[CachedFileEntry])

  /** None while the backend reads the header. */
  private val info = Var(Option.empty[Try[ModelFileInfo]])
  private val targetType = Var(ConversionTypes.default)
  private val family = Var("")

  /** Set when the file is already a registered model: its family is fixed. */
  private val familyFixed = Var(false)
  private val outputName = Var("")
  private val nameEdited = Var(false)
  private val rules = Var("")
  private val threads = Var("")
  private val advancedOpen = Var(false)
  private val keepIntermediate = Var(false)
  private val refusal = Var(Option.empty[String])
  private val submitted = Var(false)

  /** Whether drift dequantizes the file before sd-cli converts it. */
  private val needsDequantization: Signal[Boolean] = info.signal.map(
    _.exists(_.toOption.exists(i => i.comfyQuant.isDefined || i.scaledFp8))
  )

  /** Opens the modal on `file`, its family taken from the first registered
    * model referencing it when there is one.
    */
  def show(file: CachedFileEntry, models: List[Model]): Unit = {
    val referencing = models.find(model => file.referencedBy.contains(model.id))
    targetType.set(ConversionTypes.default)
    family.set(referencing.map(_.familyId).getOrElse(""))
    familyFixed.set(referencing.isDefined)
    nameEdited.set(false)
    outputName.set(defaultName(file, ConversionTypes.default))
    rules.set("")
    threads.set("")
    advancedOpen.set(false)
    keepIntermediate.set(false)
    refusal.set(None)
    submitted.set(false)
    info.set(None)
    entry.set(Some(file))
  }

  def hide(): Unit = entry.set(None)

  private def submit(file: CachedFileEntry): Unit = {
    refusal.set(None)
    submitted.set(true)
    conversionService.push(
      ConversionService.Command.Start(
        ConversionRequest(
          path = file.path,
          familyId = family.now().trim,
          targetType = targetType.now(),
          rules = rules.now().trim,
          outputName = outputName.now().trim,
          threads = threads.now().trim.toIntOption.filter(_ > 0),
          keepIntermediate = keepIntermediate.now()
        )
      )
    )
  }

  /** Whether the form can be sent: the header read, a family and a `.gguf`
    * name.
    */
  private val canSubmit: Signal[Boolean] = Signal
    .combine(info.signal, family.signal, outputName.signal, submitted.signal)
    .map { (loaded, familyName, name, sending) =>
      loaded.exists(_.isSuccess) &&
      familyName.trim.nonEmpty && name.trim.endsWith(".gguf") &&
      name.trim.length > ".gguf".length && !sending
    }

  private def sourceLine(file: CachedFileEntry): HtmlElement = div(
    cls := "mb-3",
    p(
      cls := "has-text-weight-bold mb-1",
      fileName(file),
      span(cls := "tag is-light ml-2", LaunchBlocker.humanBytes(file.bytes))
    ),
    child <-- info.signal.map {
      case None => p(cls := "text-secondary is-size-7", "Reading the header…")
      case Some(Failure(err)) =>
        div(
          cls := "notification is-danger is-light py-2 px-3 is-size-7",
          s"This file cannot be converted: ${err.getMessage}"
        )
      case Some(Success(loaded)) =>
        val types = loaded.bytesByType.toList
          .sortBy(-_._2)
          .map((name, bytes) => s"$name ${LaunchBlocker.humanBytes(bytes)}")
          .mkString(" · ")
        div(
          p(
            cls := "text-secondary is-size-7",
            s"${loaded.format}, ${loaded.tensorCount} tensors, ${humanParameters(loaded.parameterCount)} parameters — $types"
          ),
          // sd-cpp's converter leaves int8 tensors as they are and never
          // applies fp8 scales, so drift rewrites such a file as plain
          // floats first (`specs/25-model-conversion.md`, step 2).
          {
            val intermediate =
              LaunchBlocker.humanBytes(loaded.parameterCount * 2)
            loaded.comfyQuant match {
              case Some(quant) =>
                div(
                  cls := "notification is-info is-light py-2 px-3 is-size-7",
                  s"ComfyUI ${
                      if (quant.format.isEmpty) "int8 (no format key)"
                      else quant.format
                    }${
                      if (quant.convrot)
                        s" convrot (groups of ${quant.groupSize})"
                      else ""
                    }: drift dequantizes it first — int8 × scale${
                      if (quant.convrot) ", rotation undone" else ""
                    } — into an F16 safetensors beside the output (about $intermediate), removed once converted unless kept (Advanced). The int8 rounding already in the file stays; nothing else is lost before the quant itself."
                )
              case None if loaded.scaledFp8 =>
                div(
                  cls := "notification is-info is-light py-2 px-3 is-size-7",
                  s"Scaled fp8: drift rescales it first — fp8 × scale — into an F16 safetensors beside the output (about $intermediate), removed once converted unless kept (Advanced). sd-cpp's converter would not apply the scales itself."
                )
              case None => emptyNode
            }
          }
        )
    }
  )

  private def typeField(): HtmlElement = div(
    cls := "field",
    label(cls := "label is-small", "Target type"),
    div(
      cls := "control",
      div(
        cls := "select is-small is-fullwidth",
        select(
          ConversionTypes.all.map(t => option(value := t.name, t.label)),
          value <-- targetType.signal,
          onChange.mapToValue --> { name =>
            targetType.set(name)
            entry.now().foreach { file =>
              if (!nameEdited.now()) outputName.set(defaultName(file, name))
            }
          }
        )
      )
    ),
    p(
      cls := "help",
      child.text <-- info.signal.combineWith(targetType.signal).map {
        case (Some(Success(loaded)), name) =>
          val estimate = ConversionTypes
            .estimateBytes(loaded.parameterCount, name)
            .map(LaunchBlocker.humanBytes)
            .getOrElse("?")
          val sourceBits =
            if (loaded.parameterCount == 0) 0.0
            else loaded.bytesByType.values.sum * 8.0 / loaded.parameterCount
          val wider = ConversionTypes
            .byName(name)
            .exists(_.bitsPerWeight > sourceBits + 0.5)
          s"About $estimate (sd-cpp keeps embeddings, norms and the in/out projections unquantized, so a little more)." +
            (if (wider)
               s" The source holds about ${sourceBits.round} bits per weight — a wider type gains no quality."
             else "")
        case _ => ""
      }
    )
  )

  private def familyField(): HtmlElement = div(
    cls := "field",
    label(cls := "label is-small", "Family"),
    div(
      cls := "control",
      child <-- familyFixed.signal.map {
        case true =>
          p(
            cls := "is-size-7",
            child.text <-- family.signal.map(name =>
              s"$name — the family of the model this file belongs to"
            )
          )
        case false =>
          input(
            cls := "input is-small",
            placeholder := "family the result joins",
            listId := "known-families",
            controlled(
              value <-- family.signal,
              onInput.mapToValue --> family
            )
          )
      }
    )
  )

  private def nameField(): HtmlElement = div(
    cls := "field",
    label(cls := "label is-small", "Output file"),
    div(
      cls := "control",
      input(
        cls := "input is-small",
        controlled(
          value <-- outputName.signal,
          onInput.mapToValue --> { name =>
            nameEdited.set(true)
            outputName.set(name)
          }
        )
      )
    ),
    p(
      cls := "help",
      child.text <-- family.signal.map(name =>
        s"Written to ~/.cache/drift/models/${
            if (name.trim.isEmpty) "<family>" else name.trim
          }/converted/, then registered as a model of the family."
      )
    )
  )

  private def advanced(): HtmlElement = div(
    cls := "mb-3",
    a(
      cls := "is-size-7",
      child.text <-- advancedOpen.signal.map(open =>
        if (open) "▾ Advanced" else "▸ Advanced"
      ),
      onClick --> (_ => advancedOpen.update(!_))
    ),
    child <-- advancedOpen.signal.map {
      case false => emptyNode
      case true  =>
        div(
          cls := "mt-2",
          div(
            cls := "field",
            label(cls := "label is-small", "Tensor type rules"),
            div(
              cls := "control",
              input(
                cls := "input is-small",
                placeholder := "regex=type,regex=type — e.g. final_layer=f16,single_blocks\\.(0|1)\\.=q8_0",
                controlled(
                  value <-- rules.signal,
                  onInput.mapToValue --> rules
                )
              )
            ),
            p(
              cls := "help",
              "sd-cpp's --tensor-type-rules: matched in order with a regex search, the first match wins over the target type."
            )
          ),
          child <-- needsDequantization.map {
            case false => emptyNode
            case true  =>
              div(
                cls := "field",
                label(
                  cls := "checkbox is-size-7",
                  input(
                    tpe := "checkbox",
                    controlled(
                      checked <-- keepIntermediate.signal,
                      onClick.mapToChecked --> keepIntermediate
                    )
                  ),
                  " Keep the dequantized safetensors — to run it on the GPU as it is and judge the dequantization apart from the quant"
                )
              )
          },
          div(
            cls := "field",
            label(cls := "label is-small", "Threads"),
            div(
              cls := "control",
              input(
                cls := "input is-small",
                tpe := "number",
                minAttr := "1",
                placeholder := "sd-cpp's default: the physical cores",
                controlled(
                  value <-- threads.signal,
                  onInput.mapToValue --> threads
                )
              )
            )
          )
        )
    }
  )

  private def modal(file: CachedFileEntry): HtmlElement = div(
    cls := "modal is-active convert-model-modal",
    ScrollLock.whileMounted,
    conversionService.inspect(file.path).recoverToTry --> { result =>
      info.set(Some(result))
    },
    conversionService.events --> Observer[ConversionService.Event] {
      case ConversionService.Event.Refused(reason) =>
        refusal.set(Some(reason))
        submitted.set(false)
      case _ => ()
    },
    // The job this modal asked for has been queued: it shows in the page's
    // job list, so the modal is done.
    conversionService.jobs.changes
      .filter(_ => submitted.now())
      .filter(jobs =>
        jobs.exists(job =>
          job.sourcePath == file.path &&
            job.outputPath.endsWith(s"/${outputName.now().trim}")
        )
      ) --> (_ => hide()),
    documentEvents(_.onKeyDown).filter(_.key == "Escape") --> (_ => hide()),
    div(cls := "modal-background", onClick --> (_ => hide())),
    div(
      cls := "modal-card",
      headerTag(
        cls := "modal-card-head",
        p(cls := "modal-card-title", "Convert to GGUF"),
        button(
          cls := "delete",
          aria.label := "close",
          onClick --> (_ => hide())
        )
      ),
      sectionTag(
        cls := "modal-card-body",
        sourceLine(file),
        typeField(),
        familyField(),
        nameField(),
        advanced(),
        child <-- refusal.signal.map {
          case Some(reason) =>
            div(
              cls := "notification is-danger is-light py-2 px-3 is-size-7",
              reason
            )
          case None => emptyNode
        }
      ),
      footerTag(
        cls := "modal-card-foot",
        button(
          cls := "button is-info",
          "Convert",
          disabled <-- canSubmit.map(!_),
          onClick --> (_ => submit(file))
        ),
        button(cls := "button", "Cancel", onClick --> (_ => hide()))
      )
    )
  )

  lazy val element: HtmlElement = div(
    child <-- entry.signal.map {
      case None       => emptyNode
      case Some(file) => modal(file)
    }
  )
}

object ConvertModelModal {
  private val minAttr =
    htmlAttr("min", com.raquo.laminar.codecs.StringAsIsCodec)

  /** "Sidecar name — file.ext" and "sub/dir/file.ext" both end in the filename,
    * whatever kind of row this is.
    */
  def fileName(file: CachedFileEntry): String =
    file.label.split(" — ").last.split('/').last

  /** `<source base>-<type>.gguf`, the type spelled as a file name usually is.
    */
  def defaultName(file: CachedFileEntry, typeName: String): String = {
    val name = fileName(file)
    val base = name.lastIndexOf('.') match {
      case -1    => name
      case index => name.take(index)
    }
    s"$base-${typeName.toUpperCase}.gguf"
  }

  def humanParameters(count: Long): String =
    if (count >= 1_000_000_000L) f"${count / 1e9}%.1fB"
    else if (count >= 1_000_000L) f"${count / 1e6}%.0fM"
    else count.toString
}
