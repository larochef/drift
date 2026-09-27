package drift.backend.conversion

import drift.shared.ConversionRequest

import java.nio.file.Path

/** The `sd-cli -M convert` command line (`specs/25-model-conversion.md`). The
  * source goes in with `-m`, which keeps its tensor names verbatim, so the
  * output is a drop-in for the source under whatever flag the source used
  * (`--diffusion-model`, `--vae`, `--llm`, …) — upstream converts single
  * component files the same way.
  */
private[conversion] object SdCppConvert {

  def command(
      cli: Path,
      source: Path,
      output: Path,
      request: ConversionRequest
  ): List[String] =
    List(
      cli.toString,
      "-M",
      "convert",
      "-m",
      source.toString,
      "-o",
      output.toString,
      "--type",
      request.targetType
    ) ++
      Option(request.rules.trim)
        .filter(_.nonEmpty)
        .toList
        .flatMap(rules => List("--tensor-type-rules", rules)) ++
      request.threads.toList.flatMap(threads => List("-t", threads.toString))
}
