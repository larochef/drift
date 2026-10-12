package drift.frontend.generate

import com.raquo.laminar.api.L.*

/** A form field whose value comes in layers
  * (`specs/49-lora-sampling-settings.md`, `specs/16-parameter-resolution.md`):
  * the user's when they gave one, else the default — the session's, with the
  * selected LoRAs' settings over it. A default that changes shows at once in a
  * field the user left alone and never replaces what they wrote.
  */
final class LayeredField {

  private val layers = Var(LayeredField.Layers("", None))

  /** What the field shows. Writing it is the user's doing — typing, or a recipe
    * laid down: the value is theirs from then on.
    */
  val value: Var[String] =
    layers.zoomLazy(_.shown)((held, written) => held.copy(own = Some(written)))

  /** The default under whatever the user wrote. */
  def default(value: String): Unit = layers.update(_.copy(default = value))

  /** Lets go of what the user wrote: the field shows its default again. */
  def release(): Unit = layers.update(_.copy(own = None))

  /** What the field shows becomes its default, and nobody's own. */
  def settle(): Unit =
    layers.update(held => LayeredField.Layers(held.shown, None))

  /** What the field shows becomes the user's, whoever put it there. */
  def hold(): Unit = layers.update(held => held.copy(own = Some(held.shown)))
}

object LayeredField {
  final private case class Layers(default: String, own: Option[String]) {
    def shown: String = own.getOrElse(default)
  }
}
