package drift.runner.text.jinja

import drift.runner.text.jinja.Value.Sequence

import scala.collection.mutable

final class TemplateException(message: String) extends RuntimeException(message)

/** A value in a template, with Python's semantics: chat templates are written
  * for Jinja2 on Python, and transformers renders them there, so truthiness,
  * `str()`, `repr()`, equality and the string methods follow Python.
  */
enum Value {

  /** A missing variable, attribute or key: prints as nothing, is false and
    * iterates as empty, but reading an attribute of it fails.
    */
  case Undefined(name: String)
  case NoneValue
  case Bool(value: Boolean)
  case Integer(value: Long)
  case Real(value: Double)
  case Text(value: String)

  /** A list. */
  case Items(values: Vector[Value])

  /** A tuple: a list in all but its repr, `('a', 1)` (`items()`, `dictsort`).
    */
  case Tuple(values: Vector[Value])

  /** A dict, in insertion order. */
  case Dict(entries: mutable.LinkedHashMap[String, Value])

  /** `namespace(...)`: the one mutable object, set by `{% set ns.x = … %}`. */
  case Namespace(fields: mutable.LinkedHashMap[String, Value])

  /** A macro, a global function or a bound method. */
  case Function(name: String, call: (Seq[Value], Map[String, Value]) => Value)

  def truthy: Boolean = this match {
    case Undefined(_)                  => false
    case NoneValue                     => false
    case Bool(value)                   => value
    case Integer(value)                => value != 0
    case Real(value)                   => value != 0
    case Text(value)                   => value.nonEmpty
    case Items(values)                 => values.nonEmpty
    case Tuple(values)                 => values.nonEmpty
    case Dict(entries)                 => entries.nonEmpty
    case Namespace(_) | Function(_, _) => true
  }

  /** Python's `str()`, which is what `{{ }}` prints. */
  def show: String = this match {
    case Undefined(_) => ""
    case Text(value)  => value
    case other        => other.repr
  }

  /** Python's `repr()`. */
  def repr: String = this match {
    case Undefined(_)   => ""
    case NoneValue      => "None"
    case Bool(value)    => if (value) "True" else "False"
    case Integer(value) => value.toString
    case Real(value)    => Value.pythonFloat(value)
    case Text(value)    => Value.pythonQuote(value)
    case Items(values)  => values.map(_.repr).mkString("[", ", ", "]")
    case Tuple(values)  =>
      values
        .map(_.repr)
        .mkString("(", ", ", if (values.size == 1) ",)" else ")")
    case Dict(entries) =>
      entries
        .map((k, v) => s"${Value.pythonQuote(k)}: ${v.repr}")
        .mkString("{", ", ", "}")
    case Namespace(f) =>
      f.map((k, v) => s"$k=${v.repr}").mkString("<Namespace ", ", ", ">")
    case Function(name, _) => s"<function $name>"
  }

  def isUndefined: Boolean = this.isInstanceOf[Undefined]

  /** What `for` walks: a dict's keys, a string's characters. */
  def iterate: Vector[Value] = this match {
    case Undefined(_)     => Vector.empty
    case Sequence(values) => values
    case Dict(entries)    => entries.keys.map(Text(_)).toVector
    case Text(value)      =>
      value
        .codePoints()
        .toArray
        .map(c => Text(new String(Character.toChars(c))))
        .toVector
    case other =>
      throw new TemplateException(s"'${other.typeName}' object is not iterable")
  }

  def typeName: String = this match {
    case Undefined(_)   => "Undefined"
    case NoneValue      => "NoneType"
    case Bool(_)        => "bool"
    case Integer(_)     => "int"
    case Real(_)        => "float"
    case Text(_)        => "str"
    case Items(_)       => "list"
    case Tuple(_)       => "tuple"
    case Dict(_)        => "dict"
    case Namespace(_)   => "Namespace"
    case Function(_, _) => "function"
  }
}

object Value {

  /** A list or a tuple, read alike. */
  object Sequence {
    def unapply(value: Value): Option[Vector[Value]] = value match {
      case Items(values) => Some(values)
      case Tuple(values) => Some(values)
      case _             => None
    }
  }

  def text(value: String): Value = Text(value)

  /** A JSON value (messages, tools) as a template value. */
  def fromJson(json: ujson.Value): Value = json match {
    case ujson.Null    => NoneValue
    case ujson.Bool(b) => Bool(b)
    case ujson.Num(n)  =>
      if (n == math.rint(n) && math.abs(n) < 9e15) Integer(n.toLong)
      else Real(n)
    case ujson.Str(s) => Text(s)
    case ujson.Arr(a) => Items(a.map(fromJson).toVector)
    // ujson's own ordered map: `.map` on it would return an unordered one
    case ujson.Obj(o) =>
      Dict(
        mutable.LinkedHashMap.from(o.iterator.map((k, v) => k -> fromJson(v)))
      )
  }

  /** Python's float repr for the common cases: `1.0`, `0.5`, `1e-05`. */
  def pythonFloat(value: Double): String =
    if (value.isNaN) "nan"
    else if (value.isInfinite) (if (value > 0) "inf" else "-inf")
    else if (value == math.rint(value) && math.abs(value) < 1e16) f"$value%.1f"
    else {
      val text =
        java.math.BigDecimal.valueOf(value).stripTrailingZeros().toString
      if (!text.contains("E")) text
      else {
        // Java 1.0E-5 → Python 1e-05
        val Array(mantissa, exponent) = text.split("E")
        val e = exponent.toInt
        s"${mantissa}e${if (e < 0) "-" else "+"}${f"${math.abs(e)}%02d"}"
      }
    }

  /** Python's string repr: single quotes unless the text has one and no double
    * quote.
    */
  def pythonQuote(value: String): String = {
    val quote = if (value.contains('\'') && !value.contains('"')) '"' else '\''
    val body = value.flatMap {
      case '\\'            => "\\\\"
      case '\n'            => "\\n"
      case '\r'            => "\\r"
      case '\t'            => "\\t"
      case c if c == quote => s"\\$c"
      case c if c < ' '    => f"\\x${c.toInt}%02x"
      case c               => c.toString
    }
    s"$quote$body$quote"
  }

  /** Python's `==`. */
  def equal(a: Value, b: Value): Boolean = (a, b) match {
    case (Integer(x), Real(y))      => x.toDouble == y
    case (Real(x), Integer(y))      => x == y.toDouble
    case (Bool(x), Integer(y))      => (if (x) 1L else 0L) == y
    case (Integer(x), Bool(y))      => x == (if (y) 1L else 0L)
    case (Sequence(x), Sequence(y)) =>
      x.size == y.size && x.zip(y).forall(equal)
    case (Dict(x), Dict(y)) =>
      x.keySet == y.keySet && x.forall((k, v) => equal(v, y(k)))
    case (Undefined(_), Undefined(_)) => true
    case _                            => a == b
  }

  /** Python's ordering, for `<` and sorting. */
  def compare(a: Value, b: Value): Int = (a, b) match {
    case (Text(x), Text(y))         => x.compareTo(y)
    case (Sequence(x), Sequence(y)) =>
      x.zip(y)
        .map(compare)
        .find(_ != 0)
        .getOrElse(java.lang.Integer.compare(x.size, y.size))
    case _ =>
      (number(a), number(b)) match {
        case (Some(x), Some(y)) => java.lang.Double.compare(x, y)
        case _                  =>
          throw new TemplateException(
            s"'<' not supported between '${a.typeName}' and '${b.typeName}'"
          )
      }
  }

  def number(value: Value): Option[Double] = value match {
    case Integer(n) => Some(n.toDouble)
    case Real(n)    => Some(n)
    case Bool(b)    => Some(if (b) 1 else 0)
    case _          => None
  }

  /** `json.dumps(value, ensure_ascii=False, indent=indent)`: transformers'
    * `tojson`, which (unlike Jinja's own) does not escape HTML characters.
    */
  def toJson(value: Value, indent: Option[Int]): String = {
    val out = new StringBuilder
    def string(s: String): Unit = {
      out += '"'
      s.foreach {
        case '"'          => out ++= "\\\""
        case '\\'         => out ++= "\\\\"
        case '\n'         => out ++= "\\n"
        case '\r'         => out ++= "\\r"
        case '\t'         => out ++= "\\t"
        case '\b'         => out ++= "\\b"
        case '\f'         => out ++= "\\f"
        case c if c < ' ' => out ++= f"\\u${c.toInt}%04x"
        case c            => out += c
      }
      out += '"'
    }
    def write(v: Value, depth: Int): Unit = {
      def newline(level: Int) =
        indent.foreach(n => out ++= "\n" + " " * (n * level))
      val separator = if (indent.isDefined) "," else ", "
      v match {
        case NoneValue | Undefined(_) => out ++= "null"
        case Bool(b)                  => out ++= (if (b) "true" else "false")
        case Integer(n)               => out ++= n.toString
        case Real(n)                  => out ++= pythonFloat(n)
        case Text(s)                  => string(s)
        case Sequence(values)         =>
          if (values.isEmpty) out ++= "[]"
          else {
            out += '['
            values.zipWithIndex.foreach { (item, i) =>
              if (i > 0) out ++= separator
              newline(depth + 1)
              write(item, depth + 1)
            }
            newline(depth)
            out += ']'
          }
        case Dict(entries) =>
          if (entries.isEmpty) out ++= "{}"
          else {
            out += '{'
            entries.zipWithIndex.foreach { case ((key, item), i) =>
              if (i > 0) out ++= separator
              newline(depth + 1)
              string(key)
              out ++= ": "
              write(item, depth + 1)
            }
            newline(depth)
            out += '}'
          }
        case other =>
          throw new TemplateException(
            s"Object of type ${other.typeName} is not JSON serializable"
          )
      }
    }
    write(value, 0)
    out.toString
  }
}
