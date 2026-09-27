package drift.frontend.components

import drift.shared.LoraGrouping

import com.raquo.laminar.api.L.*

/** One file a browser's list offers to install: what tells it from the others,
  * and what it will cost.
  */
case class SelectedFile(key: String, sizeBytes: Option[Long] = None)

/** What is ticked in a browser's file list (`specs/33-lora-sources.md`).
  * Nothing, to begin with: a LoRA is chosen file by file (François,
  * 2026-09-18), and the ticks may cross a Civitai model's versions — the whole
  * selection installs at once, which is what makes a wan 2.2 pair one LoRA.
  */
class FileSelection {

  private val ticked = Var(List.empty[SelectedFile])
  private val _sent = Var(Option.empty[Int])

  /** In the order they were ticked. */
  val files: Signal[List[SelectedFile]] = ticked.signal

  /** How many files the last install took, until the ticks change again: the
    * browser stays open, so the bar must say what already happened.
    */
  val sent: Signal[Option[Int]] = _sent.signal

  def isTicked(key: String): Signal[Boolean] =
    ticked.signal.map(_.exists(_.key == key))

  def keys: List[String] = ticked.now().map(_.key)

  def toggle(file: SelectedFile): Unit = {
    _sent.set(None)
    ticked.update(current =>
      if (current.exists(_.key == file.key))
        current.filterNot(_.key == file.key)
      else current :+ file
    )
  }

  /** The install has gone out: the ticks go with it, the count stays. */
  def sendOff(): Unit = {
    _sent.set(Some(ticked.now().size))
    ticked.set(Nil)
  }

  /** Another model or repository is opened: its own files start unticked. */
  def clear(): Unit = {
    _sent.set(None)
    ticked.set(Nil)
  }

  /** The bar over the list, when the list installs: every file list builds it
    * the same way, from the same selection (`specs/33-lora-sources.md`).
    */
  def bar(
      candidates: Signal[List[InstalledItem]],
      onInstall: Option[(List[String], LoraGrouping) => Unit],
      what: String = "weight files"
  ): Mod[HtmlElement] = onInstall.map(install =>
    InstallBar(
      selection = this,
      candidates = candidates,
      onInstall = grouping => install(keys, grouping),
      what = what
    ).element
  )

  /** The box a tickable row shows. */
  def box(key: String, ticked: Set[String]): String =
    if (ticked.contains(key)) "☑" else "☐"
}
