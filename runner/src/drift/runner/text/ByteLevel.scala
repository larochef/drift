package drift.runner.text

/** GPT-2's byte-level alphabet: each byte stands for one printable character,
  * so a byte-level vocabulary's tokens are strings of these characters. The
  * printable ASCII and Latin-1 bytes stand for themselves; the rest (controls,
  * space, …) for the characters from U+0100 on, in byte order — so a space is
  * `Ġ` (U+0120).
  */
object ByteLevel {

  /** The character standing for each byte. */
  val characterOf: Array[Char] = {
    val printable =
      (('!'.toInt to '~'.toInt) ++ ('¡'.toInt to '¬'.toInt) ++ ('®'.toInt to 'ÿ'.toInt)).toSet
    var next = 256
    Array.tabulate(256) { byte =>
      if (printable(byte)) byte.toChar
      else {
        val stand = next.toChar
        next += 1
        stand
      }
    }
  }

  /** The byte each character stands for; -1 for characters outside the
    * alphabet.
    */
  val byteOf: Map[Char, Int] = characterOf.zipWithIndex.toMap

  /** A string's UTF-8 bytes, in the alphabet. */
  def encode(text: String): String = {
    val bytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val out = new java.lang.StringBuilder(bytes.length)
    bytes.foreach(byte => out.append(characterOf(byte & 0xff)))
    out.toString
  }
}
