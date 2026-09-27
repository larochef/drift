package drift.runner

import java.nio.file.{Path, Paths}

import drift.runner.formats.SafetensorsModel

/** The golden files `runner/fixtures/generate.py` writes into the test
  * resources.
  */
object Fixtures {

  def path(relative: String): Path =
    Paths.get(
      Option(getClass.getResource(s"/fixtures/$relative"))
        .getOrElse(throw new IllegalStateException(s"no fixture $relative"))
        .toURI
    )

  /** Opens a safetensors fixture for the length of `body`. */
  def withSafetensors[A](relative: String)(body: SafetensorsModel => A): A = {
    val model = SafetensorsModel.open(path(relative))
    try body(model)
    finally model.close()
  }
}
