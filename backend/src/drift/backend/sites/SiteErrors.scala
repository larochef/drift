package drift.backend.sites

/** What went wrong in a failed request to another site (`specs/24`), in its
  * root cause's words — the client wraps "Read timed out" in "Exception when
  * sending request: GET …" — or the cause's type when it has none.
  */
def rootReason(error: Throwable): String = {
  val root =
    Iterator.iterate(error)(_.getCause).takeWhile(_ != null).toList.last
  Option(root.getMessage)
    .filter(_.nonEmpty)
    .getOrElse(root.getClass.getSimpleName)
}
