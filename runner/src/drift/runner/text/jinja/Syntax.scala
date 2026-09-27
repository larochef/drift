package drift.runner.text.jinja

import scala.collection.mutable

/** A template's tree: the Jinja subset chat templates use. */
enum Node {
  case Literal(text: String)
  case Output(expression: Expression)
  case If(branches: Seq[(Expression, Seq[Node])], otherwise: Seq[Node])
  case For(
      targets: Seq[String],
      iterable: Expression,
      condition: Option[Expression],
      body: Seq[Node],
      otherwise: Seq[Node]
  )
  case Assign(targets: Seq[String], value: Expression)
  case AssignAttribute(name: String, attribute: String, value: Expression)
  case AssignBlock(name: String, body: Seq[Node])
  case Macro(
      name: String,
      parameters: Seq[(String, Option[Expression])],
      body: Seq[Node]
  )
  case Break
  case Continue
}

enum Expression {
  case Constant(value: Value)
  case Name(name: String)
  case Attribute(target: Expression, name: String)
  case Item(target: Expression, key: Expression)
  case Slice(
      target: Expression,
      start: Option[Expression],
      stop: Option[Expression],
      step: Option[Expression]
  )
  case Call(
      function: Expression,
      arguments: Seq[Expression],
      keywords: Seq[(String, Expression)]
  )
  case Filter(
      target: Expression,
      name: String,
      arguments: Seq[Expression],
      keywords: Seq[(String, Expression)]
  )
  case Test(
      target: Expression,
      name: String,
      arguments: Seq[Expression],
      negated: Boolean
  )
  case Unary(operator: String, operand: Expression)
  case Binary(operator: String, left: Expression, right: Expression)
  case Conditional(
      condition: Expression,
      whenTrue: Expression,
      whenFalse: Option[Expression]
  )
  case ListOf(items: Seq[Expression])
  case TupleOf(items: Seq[Expression])
  case DictOf(entries: Seq[(Expression, Expression)])
}

/** A piece of template source: text, or a tag's tokens. */
private enum Segment {
  case Text(text: String)
  case Print(tokens: Vector[Token], at: Int)
  case Statement(tokens: Vector[Token], at: Int)
}

private enum Token {
  case Word(name: String)
  case Str(value: String)
  case Int(value: Long)
  case Float(value: Double)
  case Symbol(symbol: String)
}

/** Parses a template the way transformers' environment reads it: `trim_blocks`
  * (a newline right after a block or comment tag goes) and `lstrip_blocks`
  * (spaces and tabs from a line's start to a block tag go), plus the `-`
  * markers.
  */
object Syntax {

  def parse(source: String): Seq[Node] = {
    val parser = new Parser(segments(source), source)
    val nodes = parser.nodes(Set.empty)
    parser.expectEnd()
    nodes
  }

  private def fail(source: String, at: Int, message: String): Nothing = {
    val line = source.take(at).count(_ == '\n') + 1
    throw new TemplateException(s"template line $line: $message")
  }

  private def segments(source: String): Vector[Segment] = {
    val out = Vector.newBuilder[Segment]
    var position = 0
    var stripNext = false // a '-' on the previous tag's right
    var trimNewline = false // trim_blocks after the previous block tag
    while (position < source.length) {
      val open = Seq("{{", "{%", "{#")
        .map(o => source.indexOf(o, position))
        .filter(_ >= 0)
        .minOption
      val start = open.getOrElse(source.length)
      var text = source.substring(position, start)
      if (stripNext) text = text.dropWhile(_.isWhitespace)
      else if (trimNewline)
        text =
          if (text.startsWith("\r\n")) text.drop(2) else text.stripPrefix("\n")
      stripNext = false
      trimNewline = false
      if (open.isEmpty) {
        out += Segment.Text(text)
        position = source.length
      } else {
        val kind = source.charAt(start + 1)
        val marker =
          if (start + 2 < source.length) source.charAt(start + 2) else ' '
        val block = kind == '%' || kind == '#'
        if (marker == '-') text = text.reverse.dropWhile(_.isWhitespace).reverse
        else if (block && marker != '+') {
          // lstrip_blocks: only when the tag starts its line in the source
          var run = start
          while (
            run > 0 && (source
              .charAt(run - 1) == ' ' || source.charAt(run - 1) == '\t')
          ) run -= 1
          if (run == 0 || source.charAt(run - 1) == '\n') {
            val strip = math.min(
              start - run,
              text.reverse.takeWhile(c => c == ' ' || c == '\t').length
            )
            text = text.dropRight(strip)
          }
        }
        out += Segment.Text(text)
        val contentStart =
          start + 2 + (if (marker == '-' || marker == '+') 1 else 0)
        val close =
          if (kind == '#') source.indexOf("#}", contentStart)
          else closing(source, contentStart, kind)
        if (close < 0) fail(source, start, "unclosed tag")
        val rightMarker = source.charAt(close - 1)
        val contentEnd =
          if (rightMarker == '-' && close - 1 >= contentStart) close - 1
          else close
        stripNext = rightMarker == '-'
        trimNewline = block && !stripNext
        kind match {
          case '{' =>
            out += Segment.Print(
              tokens(source, contentStart, contentEnd),
              start
            )
          case '%' =>
            out += Segment.Statement(
              tokens(source, contentStart, contentEnd),
              start
            )
          case _ => ()
        }
        position = close + 2
      }
    }
    out.result()
  }

  /** The `}}` or `%}` ending a tag, skipping string literals. */
  private def closing(source: String, from: Int, kind: Char): Int = {
    val close = if (kind == '{') "}}" else "%}"
    var i = from
    var quote = 0.toChar
    while (i < source.length - 1) {
      val c = source.charAt(i)
      if (quote != 0) {
        if (c == '\\') i += 1
        else if (c == quote) quote = 0
      } else if (c == '"' || c == '\'') quote = c
      else if (source.startsWith(close, i)) return i
      i += 1
    }
    -1
  }

  private val symbols =
    Seq(
      "**",
      "//",
      "==",
      "!=",
      "<=",
      ">=",
      "+",
      "-",
      "*",
      "/",
      "%",
      "~",
      "<",
      ">",
      "=",
      "|",
      ".",
      ",",
      ":",
      "(",
      ")",
      "[",
      "]",
      "{",
      "}"
    )

  private def tokens(source: String, from: Int, to: Int): Vector[Token] = {
    val out = Vector.newBuilder[Token]
    var i = from
    while (i < to) {
      val c = source.charAt(i)
      if (c.isWhitespace) i += 1
      else if (c.isLetter || c == '_') {
        val start = i
        while (
          i < to && (source.charAt(i).isLetterOrDigit || source.charAt(
            i
          ) == '_')
        ) i += 1
        out += Token.Word(source.substring(start, i))
      } else if (c.isDigit) {
        val start = i
        while (i < to && (source.charAt(i).isDigit || source.charAt(i) == '_'))
          i += 1
        if (
          i + 1 < to && source.charAt(i) == '.' && source.charAt(i + 1).isDigit
        ) {
          i += 1
          while (i < to && source.charAt(i).isDigit) i += 1
          out += Token.Float(
            source.substring(start, i).replace("_", "").toDouble
          )
        } else
          out += Token.Int(source.substring(start, i).replace("_", "").toLong)
      } else if (c == '"' || c == '\'') {
        val value = new StringBuilder
        i += 1
        while (i < to && source.charAt(i) != c) {
          if (source.charAt(i) == '\\' && i + 1 < to) {
            i += 1
            source.charAt(i) match {
              case 'n' => value += '\n'
              case 't' => value += '\t'
              case 'r' => value += '\r'
              case '0' => value += '\u0000'
              case 'u' =>
                value += Integer
                  .parseInt(source.substring(i + 1, i + 5), 16)
                  .toChar
                i += 4
              case other => value += other
            }
          } else value += source.charAt(i)
          i += 1
        }
        if (i >= to) fail(source, from, "unclosed string")
        i += 1
        out += Token.Str(value.toString)
      } else
        symbols.find(source.startsWith(_, i)) match {
          case Some(symbol) =>
            out += Token.Symbol(symbol)
            i += symbol.length
          case None => fail(source, i, s"unexpected character '$c'")
        }
    }
    out.result()
  }

  final private class Parser(segments: Vector[Segment], source: String) {
    private var index = 0

    def expectEnd(): Unit =
      if (index < segments.size) segments(index) match {
        case Segment.Statement(tokens, at) =>
          fail(source, at, s"unexpected ${tokens.headOption.getOrElse("tag")}")
        case _ => ()
      }

    private def keyword(segment: Segment): Option[String] = segment match {
      case Segment.Statement(Token.Word(word) +: _, _) => Some(word)
      case _                                           => None
    }

    /** Nodes up to a statement starting with one of `ends` (left unread). */
    def nodes(ends: Set[String]): Seq[Node] = {
      val out = Vector.newBuilder[Node]
      while (index < segments.size && !keyword(segments(index)).exists(ends)) {
        segments(index) match {
          case Segment.Text(text) =>
            if (text.nonEmpty) out += Node.Literal(text)
            index += 1
          case Segment.Print(tokens, at) =>
            index += 1
            val reader = new ExpressionReader(tokens, source, at)
            out += Node.Output(reader.expression())
            reader.expectEnd()
          case Segment.Statement(tokens, at) =>
            index += 1
            out += statement(tokens, at)
        }
      }
      out.result()
    }

    /** Consumes the statement that ended a block, returning its tokens. */
    private def closer(name: String, at: Int): (String, Vector[Token]) = {
      if (index >= segments.size) fail(source, at, s"'$name' is never closed")
      val Segment.Statement(tokens, _) = segments(index): @unchecked
      index += 1
      (keyword(segments(index - 1)).get, tokens)
    }

    private def statement(tokens: Vector[Token], at: Int): Node = {
      val reader = new ExpressionReader(tokens.drop(1), source, at)
      val node = keyword(Segment.Statement(tokens, at))
        .getOrElse(fail(source, at, "empty tag")) match {
        case "if" =>
          val branches =
            mutable.ArrayBuffer((reader.expression(), Seq.empty[Node]))
          var otherwise = Seq.empty[Node]
          var done = false
          while (!done) {
            val body = nodes(Set("elif", "else", "endif"))
            branches(branches.size - 1) = (branches.last._1, body)
            closer("if", at) match {
              case ("elif", t) =>
                branches += ((
                  new ExpressionReader(t.drop(1), source, at).expression(),
                  Nil
                ))
              case ("else", _) =>
                otherwise = nodes(Set("endif"))
                closer("if", at)
                done = true
              case _ => done = true
            }
          }
          Node.If(branches.toSeq, otherwise)
        case "for" =>
          val targets = reader.names()
          reader.expectWord("in")
          val iterable = reader.expression(conditional = false)
          val condition =
            Option.when(reader.acceptWord("if"))(reader.expression())
          val body = nodes(Set("else", "endfor"))
          val otherwise = closer("for", at) match {
            case ("else", _) =>
              val rest = nodes(Set("endfor"))
              closer("for", at)
              rest
            case _ => Nil
          }
          Node.For(targets, iterable, condition, body, otherwise)
        case "set" =>
          val targets = reader.names()
          if (reader.accept(".")) {
            val attribute = reader.name()
            reader.expect("=")
            Node.AssignAttribute(targets.head, attribute, reader.expression())
          } else if (reader.accept("="))
            Node.Assign(targets, reader.expression())
          else {
            val body = nodes(Set("endset"))
            closer("set", at)
            Node.AssignBlock(targets.head, body)
          }
        case "macro" =>
          val name = reader.name()
          reader.expect("(")
          val parameters =
            mutable.ArrayBuffer.empty[(String, Option[Expression])]
          while (!reader.accept(")")) {
            val parameter = reader.name()
            parameters += parameter -> Option.when(reader.accept("="))(
              reader.expression()
            )
            reader.accept(",")
          }
          val body = nodes(Set("endmacro"))
          closer("macro", at)
          Node.Macro(name, parameters.toSeq, body)
        case "break"    => Node.Break
        case "continue" => Node.Continue
        case other      => fail(source, at, s"'$other' is not supported")
      }
      reader.expectEnd()
      node
    }
  }

  /** Expressions, by Jinja's precedence: conditional, or, and, not,
    * comparisons, `~`, `+ -`, `* / // %`, `**`, unary, then postfix (`.`, `[]`,
    * calls) with filters and tests binding tightest.
    */
  final private class ExpressionReader(
      tokens: Vector[Token],
      source: String,
      at: Int
  ) {
    private var index = 0

    private def peek: Option[Token] = tokens.lift(index)
    private def fail(message: String): Nothing =
      Syntax.fail(source, at, message)

    def expectEnd(): Unit =
      if (index < tokens.size) fail(s"unexpected ${tokens(index)}")

    def accept(symbol: String): Boolean =
      if (peek.contains(Token.Symbol(symbol))) { index += 1; true }
      else false

    def expect(symbol: String): Unit = if (!accept(symbol))
      fail(s"expected '$symbol' at ${peek.getOrElse("end")}")

    def acceptWord(word: String): Boolean =
      if (peek.contains(Token.Word(word))) { index += 1; true }
      else false

    def expectWord(word: String): Unit =
      if (!acceptWord(word)) fail(s"expected '$word'")

    def name(): String = peek match {
      case Some(Token.Word(word)) =>
        index += 1
        word
      case other => fail(s"expected a name, found ${other.getOrElse("end")}")
    }

    /** `a` or `a, b` (loop and set targets). */
    def names(): Seq[String] = {
      val first = name()
      val rest = mutable.ArrayBuffer.empty[String]
      while (accept(",")) rest += name()
      first +: rest.toSeq
    }

    def expression(conditional: Boolean = true): Expression = {
      val value = or()
      if (conditional && acceptWord("if")) {
        val condition = or()
        val otherwise = Option.when(acceptWord("else"))(expression())
        Expression.Conditional(condition, value, otherwise)
      } else value
    }

    private def or(): Expression = {
      var left = and()
      while (acceptWord("or")) left = Expression.Binary("or", left, and())
      left
    }

    private def and(): Expression = {
      var left = not()
      while (acceptWord("and")) left = Expression.Binary("and", left, not())
      left
    }

    private def not(): Expression =
      if (acceptWord("not")) Expression.Unary("not", not()) else comparison()

    private def comparison(): Expression = {
      var left = concatenation()
      var going = true
      while (going) {
        val operator = peek match {
          case Some(
                Token.Symbol(s @ ("==" | "!=" | "<" | ">" | "<=" | ">="))
              ) =>
            index += 1
            Some(s)
          case Some(Token.Word("in")) =>
            index += 1
            Some("in")
          case Some(Token.Word("not"))
              if tokens.lift(index + 1).contains(Token.Word("in")) =>
            index += 2
            Some("not in")
          case _ => None
        }
        operator match {
          case Some(op) => left = Expression.Binary(op, left, concatenation())
          case None     => going = false
        }
      }
      left
    }

    private def concatenation(): Expression = {
      var left = additive()
      while (accept("~")) left = Expression.Binary("~", left, additive())
      left
    }

    private def additive(): Expression = {
      var left = multiplicative()
      var going = true
      while (going) {
        if (accept("+")) left = Expression.Binary("+", left, multiplicative())
        else if (accept("-"))
          left = Expression.Binary("-", left, multiplicative())
        else going = false
      }
      left
    }

    private def multiplicative(): Expression = {
      var left = power()
      var going = true
      while (going) {
        val operator = Seq("*", "//", "/", "%").find(accept)
        operator match {
          case Some(op) => left = Expression.Binary(op, left, power())
          case None     => going = false
        }
      }
      left
    }

    private def power(): Expression = {
      val left = unary()
      if (accept("**")) Expression.Binary("**", left, unary()) else left
    }

    private def unary(): Expression =
      if (accept("-")) Expression.Unary("-", unary())
      else if (accept("+")) unary()
      else filtered(postfix(primary()))

    private def arguments(): (Seq[Expression], Seq[(String, Expression)]) = {
      val positional = mutable.ArrayBuffer.empty[Expression]
      val keywords = mutable.ArrayBuffer.empty[(String, Expression)]
      while (!accept(")")) {
        (peek, tokens.lift(index + 1)) match {
          case (Some(Token.Word(key)), Some(Token.Symbol("="))) =>
            index += 2
            keywords += key -> expression()
          case _ => positional += expression()
        }
        if (!peek.contains(Token.Symbol(")"))) expect(",")
      }
      (positional.toSeq, keywords.toSeq)
    }

    private def postfix(start: Expression): Expression = {
      var value = start
      var going = true
      while (going) {
        if (accept(".")) value = Expression.Attribute(value, name())
        else if (accept("[")) {
          def part(): Option[Expression] =
            if (
              peek.exists(t => t == Token.Symbol(":") || t == Token.Symbol("]"))
            ) None
            else Some(expression())
          val first = part()
          if (accept(":")) {
            val stop = part()
            val step = if (accept(":")) part() else None
            value = Expression.Slice(value, first, stop, step)
          } else
            value =
              Expression.Item(value, first.getOrElse(fail("empty subscript")))
          expect("]")
        } else if (accept("(")) {
          val (positional, keywords) = arguments()
          value = Expression.Call(value, positional, keywords)
        } else going = false
      }
      value
    }

    private def filtered(start: Expression): Expression = {
      var value = start
      var going = true
      while (going) {
        if (accept("|")) {
          val filter = name()
          val (positional, keywords) =
            if (accept("(")) arguments() else (Nil, Nil)
          value = Expression.Filter(value, filter, positional, keywords)
        } else if (acceptWord("is")) {
          val negated = acceptWord("not")
          val test = name()
          val positional =
            if (accept("(")) arguments()._1
            else
              peek match {
                case Some(Token.Str(_) | Token.Int(_) | Token.Float(_)) =>
                  Seq(primary())
                case _ => Nil
              }
          value = Expression.Test(value, test, positional, negated)
        } else going = false
      }
      value
    }

    private def primary(): Expression = peek match {
      case Some(Token.Str(s)) =>
        index += 1
        var text = s
        // adjacent literals join, as in Python
        while (peek.exists(_.isInstanceOf[Token.Str])) {
          text += peek.get.asInstanceOf[Token.Str].value
          index += 1
        }
        Expression.Constant(Value.Text(text))
      case Some(Token.Int(n)) =>
        index += 1
        Expression.Constant(Value.Integer(n))
      case Some(Token.Float(n)) =>
        index += 1
        Expression.Constant(Value.Real(n))
      case Some(Token.Word("true" | "True")) =>
        index += 1
        Expression.Constant(Value.Bool(true))
      case Some(Token.Word("false" | "False")) =>
        index += 1
        Expression.Constant(Value.Bool(false))
      case Some(Token.Word("none" | "None")) =>
        index += 1
        Expression.Constant(Value.NoneValue)
      case Some(Token.Word(word)) =>
        index += 1
        Expression.Name(word)
      case Some(Token.Symbol("(")) =>
        index += 1
        if (accept(")")) Expression.TupleOf(Nil)
        else {
          val first = expression()
          if (accept(")")) first
          else {
            val items = mutable.ArrayBuffer(first)
            while (accept(",") && !peek.contains(Token.Symbol(")")))
              items += expression()
            expect(")")
            Expression.TupleOf(items.toSeq)
          }
        }
      case Some(Token.Symbol("[")) =>
        index += 1
        val items = mutable.ArrayBuffer.empty[Expression]
        while (!accept("]")) {
          items += expression()
          if (!peek.contains(Token.Symbol("]"))) expect(",")
        }
        Expression.ListOf(items.toSeq)
      case Some(Token.Symbol("{")) =>
        index += 1
        val entries = mutable.ArrayBuffer.empty[(Expression, Expression)]
        while (!accept("}")) {
          val key = expression()
          expect(":")
          entries += key -> expression()
          if (!peek.contains(Token.Symbol("}"))) expect(",")
        }
        Expression.DictOf(entries.toSeq)
      case other => fail(s"unexpected ${other.getOrElse("end of expression")}")
    }
  }
}
