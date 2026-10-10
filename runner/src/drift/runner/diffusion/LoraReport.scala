package drift.runner.diffusion

/** What a request's LoRAs came to, in the log: how many of their updates the
  * model took, and what it left — the files' tensors that are no update it
  * reads (`unread`, as `LoraFiles.open` names them) and the updates whose
  * target is no weight of the model (`unmatched`) — counted, the first of each
  * named. A file that changes nothing says so here, not in the picture.
  */
object LoraReport {

  /** How many names a line lists before it counts the rest. */
  val Named = 6

  /** For a model whose updates are its files' pairs. */
  def lines(
      loras: Seq[(Lora, Float)],
      unread: Seq[String],
      unmatched: Seq[String]
  ): Seq[String] = {
    val updates = loras.map(_._1.pairs.size).sum
    lines(loras.size, updates, updates - unmatched.size, unread, unmatched)
  }

  /** `applied` of the `updates` that `files` hold. */
  def lines(
      files: Int,
      updates: Int,
      applied: Int,
      unread: Seq[String],
      unmatched: Seq[String]
  ): Seq[String] =
    if (files == 0) Nil
    else {
      def listed(names: Seq[String]) =
        names.take(Named).mkString(", ") +
          (if (names.size > Named) s" … and ${names.size - Named} more" else "")
      Seq(
        s"LoRA: $applied of $updates updates applied ($files file${
            if (files == 1) "" else "s"
          })"
      ) ++
        Option.when(applied == 0)(
          "[WARN] LoRA left unapplied: none of the updates names a weight of this model — the file is for another model, or names its weights a way the runner does not read"
        ) ++
        Option.when(unmatched.nonEmpty)(
          s"[WARN] LoRA left unapplied: ${unmatched.size} update${
              if (unmatched.size == 1) "" else "s"
            } of no weight of this model: ${listed(unmatched)}"
        ) ++
        Option.when(unread.nonEmpty)(
          s"[WARN] LoRA left unapplied: ${unread.size} tensor${
              if (unread.size == 1) "" else "s"
            } that are no update the runner reads: ${listed(unread)}"
        )
    }

  /** Who else hears a report's warnings (the server, for the job under way),
    * each without its `[WARN]` mark.
    */
  @volatile var listener: String => Unit = _ => ()

  private val Mark = "[WARN] "

  /** `lines` printed, the warnings among them told to `listener`. */
  def told(lines: Seq[String]): Unit = lines.foreach { line =>
    println(line)
    if (line.startsWith(Mark)) listener(line.stripPrefix(Mark))
  }

  /** `lines`, printed. */
  def print(
      loras: Seq[(Lora, Float)],
      unread: Seq[String],
      unmatched: Seq[String]
  ): Unit = told(lines(loras, unread, unmatched))
}
