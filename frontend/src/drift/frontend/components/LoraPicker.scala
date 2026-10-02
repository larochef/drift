package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** Picks LoRAs of one architecture and their strengths — the generation form's
  * picker, and a run configuration's default LoRAs
  * (`specs/28-configuration-loras.md`), so search, strength and trigger words
  * behave the same in both. The selection belongs to the caller, which reads
  * it: ids and strengths-as-typed in two Vars on purpose, so a strength
  * keystroke does not rebuild the rows (and steal the input's focus).
  */
class LoraPicker(
    /** The installed LoRAs, None until they are listed. */
    collection: Signal[Option[List[Lora]]],
    architectureId: Signal[Option[String]],
    selectedIds: Var[List[String]],
    strengths: Var[Map[String, String]],
    // Unchecked by default, per François.
    includeNsfw: Var[Boolean] = Var(false),
    /** The LoRA paths the live server listed at its launch; a selected LoRA
      * whose files it does not list needs a restart. None where no server is
      * involved.
      */
    serverPaths: Signal[Option[Set[String]]] = Val(None),
    /** Inserts a trigger word into the prompt; without it the words are only
      * shown.
      */
    onTriggerWord: Option[String => Unit] = None,
    /** What the choice applies to, for the tooltips. */
    scope: String = "this generation",
    /** Beside the NSFW checkbox in the header — a run configuration's picker
      * puts the button that installs more LoRAs there.
      */
    headerAction: Modifier[HtmlElement] = emptyMod,
    /** The editor of a LoRA's sampling settings
      * (`specs/49-lora-sampling-settings.md`), opened from a selected LoRA's ⚙
      * so they are typed where they are tried; without it the mark only says
      * the LoRA carries some.
      */
    samplingEditor: Option[Lora => HtmlElement] = None
) extends Component {

  private val searchVar = Var("")

  /** The selected LoRAs whose sampling editor is open: kept here, since a row
    * is drawn again each time its LoRA is saved.
    */
  private val samplingOpen = Var(Set.empty[String])

  /** What a LoRA's settings are, on its row: a mark where they are only shown,
    * the editor's toggle where they can be typed — offered on a LoRA with none
    * yet too, which is the one that needs them.
    */
  private def samplingMark(lora: Lora): Node = {
    val colour = if (lora.sampling.nonEmpty) " is-info is-light" else ""
    val sets =
      if (lora.sampling.nonEmpty)
        s"Sets ${lora.sampling.summary.mkString(", ")}"
      else "Sets no sampling yet"
    samplingEditor match {
      case Some(_) =>
        a(
          cls := s"tag is-small ml-1$colour",
          title := s"$sets — click to edit the settings this LoRA was made for",
          "⚙ sampling",
          onClick --> (_ =>
            samplingOpen.update(open =>
              if (open.contains(lora.id)) open - lora.id else open + lora.id
            )
          )
        )
      case None if lora.sampling.nonEmpty =>
        span(cls := s"tag is-small ml-1$colour", title := sets, "⚙ sampling")
      case None => emptyNode
    }
  }

  private def add(lora: Lora): Unit = {
    selectedIds.update(ids =>
      if (ids.contains(lora.id)) ids else ids :+ lora.id
    )
    strengths.update(current =>
      if (current.contains(lora.id)) current
      else current + (lora.id -> lora.defaultStrength.toString)
    )
  }

  /** One line of the picker: what it names on the left, taking the room there
    * is and wrapping when it runs out, and its controls on the right, never
    * squeezed — so a long LoRA name never runs under a control (François,
    * 2026-09-15).
    */
  private def row(name: HtmlElement, controls: HtmlElement*): HtmlElement =
    div(
      cls := "is-flex is-align-items-center",
      styleAttr := "gap: 0.5rem;",
      div(
        styleAttr := "flex: 1 1 auto; min-width: 0; overflow-wrap: anywhere;",
        name
      ),
      div(
        cls := "is-flex is-align-items-center",
        styleAttr := "flex: 0 0 auto; gap: 0.25rem;",
        controls
      )
    )

  private def removeButton(id: String): HtmlElement =
    button(
      cls := "button is-small lora-picker-remove",
      "✕",
      title := s"Remove from $scope",
      onClick --> (_ => selectedIds.update(_.filterNot(_ == id)))
    )

  private def selectedRow(
      lora: Lora,
      paths: Option[Set[String]]
  ): HtmlElement = {
    val invisible = paths.exists(listed =>
      !lora.files.forall(file => listed.contains(lora.requestPathOf(file)))
    )
    div(
      cls := "mb-1",
      row(
        p(
          cls := "text-primary is-size-7 has-text-weight-bold mb-0",
          ProviderIcon.of(lora).map(_.amend(cls := "mr-1")),
          lora.label,
          if (lora.nsfw) span(cls := "tag is-danger is-small ml-1", "nsfw")
          else emptyNode,
          samplingMark(lora),
          if (invisible)
            span(
              cls := "tag is-warning is-small ml-1",
              title := "sd-server scans --lora-model-dir at startup and " +
                "does not list this file — installed or moved after " +
                "launch? Restart the session.",
              "⚠ needs restart"
            )
          else emptyNode
        ),
        input(
          cls := "input is-small",
          styleAttr := "width: 4.5rem;",
          typ := "number",
          stepAttr := "0.05",
          title := s"Strength for $scope",
          defaultValue := strengths
            .now()
            .getOrElse(lora.id, lora.defaultStrength.toString),
          onInput.mapToValue --> (value =>
            strengths.update(_ + (lora.id -> value))
          )
        ),
        removeButton(lora.id)
      ),
      if (lora.triggerWords.nonEmpty)
        div(
          cls := "tags mb-0",
          span(cls := "is-size-7 text-secondary mr-1", "triggers:"),
          lora.triggerWords.map(word =>
            onTriggerWord match {
              case Some(insert) =>
                span(
                  cls := "tag is-small",
                  styleAttr := "cursor: pointer;",
                  title := "Click to insert into the prompt",
                  word,
                  onClick --> (_ => insert(word))
                )
              case None => span(cls := "tag is-small", word)
            }
          )
        )
      else emptyNode,
      samplingEditor.map(editor =>
        child <-- samplingOpen.signal.map(_.contains(lora.id)).distinct.map {
          case true  => editor(lora)
          case false => emptyNode
        }
      )
    )
  }

  /** A selected LoRA the collection no longer holds: named, removable, and not
    * sent.
    */
  private def missingRow(id: String): HtmlElement =
    row(
      p(
        cls := "text-primary is-size-7 mb-0",
        span(
          cls := "tag is-warning is-small mr-1",
          title := "Uninstalled since it was chosen — it is not sent",
          "⚠ not installed"
        ),
        id
      ),
      removeButton(id)
    ).amend(cls := "mb-1")

  lazy val element: HtmlElement = div(
    cls := "box bg-table-header py-2 px-3 mb-3 lora-picker",
    // Wraps rather than overlapping when the box is narrow.
    div(
      cls := "is-flex is-align-items-center is-flex-wrap-wrap mb-1",
      styleAttr := "gap: 0.5rem;",
      label(
        cls := "label text-primary is-small mb-0",
        styleAttr := "flex: 1 1 auto;",
        "LoRAs"
      ),
      headerAction,
      label(
        cls := "checkbox is-size-7 text-secondary",
        input(
          typ := "checkbox",
          cls := "mr-1",
          checked <-- includeNsfw.signal,
          onChange.mapToChecked --> includeNsfw
        ),
        "NSFW"
      )
    ),
    children <-- selectedIds.signal.distinct
      .combineWith(collection, serverPaths.distinct)
      .map { (ids, listed, paths) =>
        val byId = listed.getOrElse(List.empty).map(l => l.id -> l).toMap
        ids.flatMap(id =>
          byId.get(id) match {
            case Some(lora) => Some(selectedRow(lora, paths))
            // Until the collection is listed, no LoRA is known to be gone.
            case None => Option.when(listed.isDefined)(missingRow(id))
          }
        )
      },
    input(
      cls := "input is-small mb-1",
      placeholder := "Search LoRAs (name, triggers, tags)…",
      value <-- searchVar.signal,
      onInput.mapToValue --> searchVar
    ),
    // Every matching LoRA, in a list of fixed height that scrolls: a long
    // collection neither stretches the form nor hides past a cut (François,
    // 2026-09-15).
    div(
      cls := "lora-picker-choices",
      children <-- collection
        .map(_.getOrElse(List.empty))
        .combineWith(
          architectureId,
          searchVar.signal,
          includeNsfw.signal,
          selectedIds.signal
        )
        .map { (all, archId, query, nsfwShown, ids) =>
          val q = query.trim.toLowerCase
          val matching = all
            .filter(l => archId.contains(l.architectureId))
            .filterNot(l => ids.contains(l.id))
            .filter(l => nsfwShown || !l.nsfw)
            .filter(l => q.isEmpty || l.searchText.contains(q))
          if (all.isEmpty)
            List(
              p(
                cls := "is-size-7 text-secondary",
                "No LoRAs installed — add them from the architecture's " +
                  "card on the Architectures page."
              )
            )
          else
            matching.map { lora =>
              row(
                p(
                  cls := "is-size-7 text-primary mb-0",
                  ProviderIcon.of(lora).map(_.amend(cls := "mr-1")),
                  lora.label,
                  if (lora.nsfw)
                    span(cls := "tag is-danger is-small ml-1", "nsfw")
                  else emptyNode
                ),
                // A tag rather than a button: the list stays a compact column
                // of names, one line each.
                span(
                  cls := "tag is-info cursor-pointer",
                  title := s"Add ${lora.label} to $scope",
                  "+ Add",
                  onClick --> (_ => add(lora))
                )
              ).amend(cls := "mb-1 lora-picker-choice")
            }
        }
    )
  )
}

object LoraPicker {

  /** A picker's selection as a configuration stores it; a strength that does
    * not parse counts as 1.
    */
  def configured(
      ids: List[String],
      strengths: Map[String, String]
  ): List[ConfiguredLora] =
    ids.map(id =>
      ConfiguredLora(
        id,
        strengths.get(id).flatMap(_.trim.toDoubleOption).getOrElse(1.0)
      )
    )

  /** A configuration's LoRAs as a picker's strengths-as-typed. */
  def strengthsOf(loras: List[ConfiguredLora]): Map[String, String] =
    loras.map(lora => lora.loraId -> lora.strength.toString).toMap
}
