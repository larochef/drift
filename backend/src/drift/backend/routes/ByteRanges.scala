package drift.backend.routes

/** The one byte range of a `Range` header that a file of `size` bytes can
  * serve (RFC 9110 §14.1.2): `bytes=first-last`, `bytes=first-` (to the end)
  * or `bytes=-count` (the last bytes), as the inclusive `(first, last)`.
  */
object ByteRanges {

  /** What a request's `Range` header asks of a file. */
  enum Request {

    /** No header, or one this server does not honour (another unit, several
      * ranges, bad syntax): the whole file, `200`.
      */
    case Whole

    /** Bytes `first` to `last` inclusive, `206`. */
    case Part(first: Long, last: Long)

    /** A range starting past the end, `416`. */
    case Unsatisfiable
  }

  private val Single = """bytes=(\d*)-(\d*)""".r

  def of(header: Option[String], size: Long): Request =
    header.map(_.trim) match {
      case Some(Single(from, to)) if from.nonEmpty || to.nonEmpty =>
        (from.toLongOption, to.toLongOption) match {
          case (None, Some(count)) =>
            if (count == 0 || size == 0) Request.Unsatisfiable
            else Request.Part(math.max(size - count, 0), size - 1)
          case (Some(first), last) if last.forall(_ >= first) =>
            if (first >= size) Request.Unsatisfiable
            else Request.Part(first, last.fold(size - 1)(math.min(_, size - 1)))
          case _ => Request.Whole
        }
      case _ => Request.Whole
    }
}
