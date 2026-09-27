package drift.runner

import scala.util.Random

/** Text that trips tokenizers: every script and whitespace the pre-tokenizer
  * regexes treat apart, contractions, digit runs, emoji sequences, text that is
  * not NFC, added tokens inside text — and seeded random mixtures of all of it.
  */
object TokenizerCorpus {

  val cases: Seq[String] = Seq(
    "",
    " ",
    "Hello, world!",
    "  leading and trailing spaces  ",
    "tabs\tand\r\nwindows\nnewlines\n\n\nand blank lines\n",
    "   \n  \n\t\t\n",
    "It's, I'M, they'LL, we'Ve, you'd — ain't",
    "12345678901234567890 3.14159 1,000,000 0x1F 2026-09-24",
    "café naïve résumé Ångström",
    "café decomposed é and composed é",
    "日本語のテキスト、句読点。漢字とカタカナ",
    "中文文本，标点符号！数字123",
    "한국어 텍스트입니다",
    "العربية نص من اليمين إلى اليسار",
    "हिन्दी देवनागरी लिपि में पाठ",
    "Ελληνικά και Русский текст",
    "emoji 😀👍🏽 👨‍👩‍👧‍👦 🇫🇷 ❤️ ✨",
    "nbsp here, ideographic　space, zero​width, line separator",
    "```scala\ndef f(x: Int): Int = x * 2\n```",
    "{\"key\": [1, 2, {\"nested\": null}], \"s\": \"a\\\"b\"}",
    "https://example.com/path?query=1&other=%20#fragment",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\nfour\n</think>\n\n4<|im_end|>",
    "<start_of_turn>user\nhi<end_of_turn>\n<|turn>model\n<|think|>",
    "<|start|>assistant<|channel|>analysis<|message|>thinking<|end|>",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
    "a" * 300,
    "Ġ literal byte-level characters ĊĠ and ▁ sentencepiece markers ▁▁",
    "mixed<|endoftext|>tokens<eos>and<bos>more"
  )

  private val alphabet: IndexedSeq[String] =
    ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".map(
      _.toString
    ) ++
      Seq(
        " ",
        " ",
        " ",
        "  ",
        "\t",
        "\n",
        "\r\n",
        " ",
        "　",
        "'",
        "'s",
        "'LL",
        ".",
        ",",
        "!",
        "?",
        "-",
        "_",
        "/",
        "\\",
        "\"",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "#",
        "@",
        "é",
        "é",
        "́",
        "ñ",
        "ß",
        "ſ",
        "K",
        "日",
        "本",
        "語",
        "中",
        "文",
        "한",
        "글",
        "ع",
        "ر",
        "ب",
        "ह",
        "ि",
        "न",
        "्",
        "Ω",
        "Ж",
        "😀",
        "👍🏽",
        "‍",
        "🇫",
        "🇷",
        "❤",
        "️",
        "​",
        " ",
        "½",
        "²",
        "٣",
        "<|im_start|>",
        "<think>",
        "<eos>",
        "<|turn>",
        "▁",
        "Ġ"
      )).toIndexedSeq

  def random(seed: Long, count: Int): Seq[String] = {
    val random = new Random(seed)
    Seq.fill(count) {
      val length = random.nextInt(60)
      Seq.fill(length)(alphabet(random.nextInt(alphabet.size))).mkString
    }
  }
}
