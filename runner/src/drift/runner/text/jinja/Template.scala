package drift.runner.text.jinja

import drift.runner.text.jinja.Value.*

import java.time.LocalDateTime
import scala.collection.mutable
import scala.util.boundary
import scala.util.boundary.break

/** A parsed template, rendered with Jinja2's semantics as transformers sets its
  * environment up: no autoescape, `loopcontrols`, and the globals
  * `raise_exception`, `strftime_now`, `range`, `namespace`. `now` is the clock
  * `strftime_now` reads.
  */
final class Template(source: String, now: () => LocalDateTime) {

  private val tree = Syntax.parse(source)

  def render(variables: Map[String, Value]): String = {
    val out = new StringBuilder
    val root = new Scope(None)
    Builtins.globals(now).foreach((k, v) => root.set(k, v))
    variables.foreach((k, v) => root.set(k, v))
    new Evaluator(root).run(tree, root, out)
    out.toString
  }
}

/** Variables visible at a point: a `for` body and a macro call open a scope, so
  * a `set` inside them does not leak out (that is what `namespace` is for).
  */
final private class Scope(parent: Option[Scope]) {
  private val variables = mutable.HashMap.empty[String, Value]
  def get(name: String): Value =
    variables.getOrElse(name, parent.fold[Value](Undefined(name))(_.get(name)))
  def set(name: String, value: Value): Unit = variables(name) = value
  def child: Scope = new Scope(Some(this))
}

private object Loop {
  case object Broken extends RuntimeException(null, null, false, false)
  case object Continued extends RuntimeException(null, null, false, false)
}

final private class Evaluator(root: Scope) {

  def run(nodes: Seq[Node], scope: Scope, out: StringBuilder): Unit =
    nodes.foreach(node => run(node, scope, out))

  private def run(node: Node, scope: Scope, out: StringBuilder): Unit =
    node match {
      case Node.Literal(text)      => out ++= text
      case Node.Output(expression) => out ++= evaluate(expression, scope).show
      case Node.If(branches, otherwise) =>
        branches.find((condition, _) =>
          evaluate(condition, scope).truthy
        ) match {
          case Some((_, body)) => run(body, scope, out)
          case None            => run(otherwise, scope, out)
        }
      case Node.For(targets, iterable, condition, body, otherwise) =>
        val items = evaluate(iterable, scope).iterate
        val kept = condition.fold(items) { c =>
          items.filter { item =>
            val probe = scope.child
            bind(probe, targets, item)
            evaluate(c, probe).truthy
          }
        }
        if (kept.isEmpty) run(otherwise, scope, out)
        else
          boundary {
            kept.zipWithIndex.foreach { (item, i) =>
              val inner = scope.child
              bind(inner, targets, item)
              inner.set("loop", loopValue(kept, i))
              try run(body, inner, out)
              catch {
                case Loop.Continued => ()
                case Loop.Broken    => break()
              }
            }
          }
      case Node.Assign(targets, value) =>
        bind(scope, targets, evaluate(value, scope))
      case Node.AssignAttribute(name, attribute, value) =>
        scope.get(name) match {
          case Namespace(fields) => fields(attribute) = evaluate(value, scope)
          case other             =>
            throw new TemplateException(
              s"cannot assign attribute on ${other.typeName} '$name'"
            )
        }
      case Node.AssignBlock(name, body) =>
        val captured = new StringBuilder
        run(body, scope, captured)
        scope.set(name, Text(captured.toString))
      case Node.Macro(name, parameters, body) =>
        scope.set(
          name,
          Function(
            name,
            (positional, keywords) => {
              val call = root.child
              parameters.zipWithIndex.foreach {
                case ((parameter, default), i) =>
                  val value = positional
                    .lift(i)
                    .orElse(keywords.get(parameter))
                    .orElse(default.map(evaluate(_, scope)))
                    .getOrElse(Undefined(parameter))
                  call.set(parameter, value)
              }
              val rendered = new StringBuilder
              run(body, call, rendered)
              Text(rendered.toString)
            }
          )
        )
      case Node.Break    => throw Loop.Broken
      case Node.Continue => throw Loop.Continued
    }

  private def bind(scope: Scope, targets: Seq[String], value: Value): Unit =
    if (targets.size == 1) scope.set(targets.head, value)
    else {
      val parts = value.iterate
      if (parts.size != targets.size)
        throw new TemplateException(
          s"cannot unpack ${parts.size} values into ${targets.size}"
        )
      targets.zip(parts).foreach(scope.set)
    }

  private def loopValue(items: Vector[Value], i: Int): Value =
    Dict(
      mutable.LinkedHashMap(
        "index" -> Integer(i + 1),
        "index0" -> Integer(i),
        "revindex" -> Integer(items.size - i),
        "revindex0" -> Integer(items.size - i - 1),
        "first" -> Bool(i == 0),
        "last" -> Bool(i == items.size - 1),
        "length" -> Integer(items.size),
        "previtem" -> (if (i > 0) items(i - 1) else Undefined("previtem")),
        "nextitem" -> (if (i + 1 < items.size) items(i + 1)
                       else Undefined("nextitem"))
      )
    )

  def evaluate(expression: Expression, scope: Scope): Value = expression match {
    case Expression.Constant(value)         => value
    case Expression.Name(name)              => scope.get(name)
    case Expression.Attribute(target, name) =>
      val value = evaluate(target, scope)
      Builtins
        .attribute(value, name)
        .getOrElse(Builtins.item(value, Text(name)))
    case Expression.Item(target, key) =>
      val value = evaluate(target, scope)
      val k = evaluate(key, scope)
      Builtins.item(value, k) match {
        case missing @ Undefined(_) =>
          k match {
            case Text(name) =>
              Builtins.attribute(value, name).getOrElse(missing)
            case _ => missing
          }
        case found => found
      }
    case Expression.Slice(target, start, stop, step) =>
      Builtins.slice(
        evaluate(target, scope),
        start.map(evaluate(_, scope)),
        stop.map(evaluate(_, scope)),
        step.map(evaluate(_, scope))
      )
    case Expression.Call(function, positional, keywords) =>
      evaluate(function, scope) match {
        case Function(_, call) =>
          call(
            positional.map(evaluate(_, scope)),
            keywords.map((k, v) => k -> evaluate(v, scope)).toMap
          )
        case Undefined(name) =>
          throw new TemplateException(s"'$name' is undefined")
        case other =>
          throw new TemplateException(
            s"'${other.typeName}' object is not callable"
          )
      }
    case Expression.Filter(target, name, positional, keywords) =>
      Builtins.filter(
        name,
        evaluate(target, scope),
        positional.map(evaluate(_, scope)),
        keywords.map((k, v) => k -> evaluate(v, scope)).toMap
      )
    case Expression.Test(target, name, positional, negated) =>
      val result = Builtins.test(
        name,
        evaluate(target, scope),
        positional.map(evaluate(_, scope))
      )
      Bool(result != negated)
    case Expression.Unary("not", operand) =>
      Bool(!evaluate(operand, scope).truthy)
    case Expression.Unary(_, operand) =>
      evaluate(operand, scope) match {
        case Integer(n) => Integer(-n)
        case Real(n)    => Real(-n)
        case other      =>
          throw new TemplateException(
            s"bad operand type for unary -: '${other.typeName}'"
          )
      }
    case Expression.Binary("and", left, right) =>
      val l = evaluate(left, scope)
      if (!l.truthy) l else evaluate(right, scope)
    case Expression.Binary("or", left, right) =>
      val l = evaluate(left, scope)
      if (l.truthy) l else evaluate(right, scope)
    case Expression.Binary(operator, left, right) =>
      Builtins.binary(operator, evaluate(left, scope), evaluate(right, scope))
    case Expression.Conditional(condition, whenTrue, whenFalse) =>
      if (evaluate(condition, scope).truthy) evaluate(whenTrue, scope)
      else whenFalse.fold[Value](Undefined("else"))(evaluate(_, scope))
    case Expression.ListOf(items) =>
      Items(items.map(evaluate(_, scope)).toVector)
    case Expression.TupleOf(items) =>
      Tuple(items.map(evaluate(_, scope)).toVector)
    case Expression.DictOf(entries) =>
      Dict(mutable.LinkedHashMap.from(entries.map { (k, v) =>
        evaluate(k, scope).show -> evaluate(v, scope)
      }))
  }
}

/** The filters, tests, methods, operators and globals chat templates use, with
  * Python's behaviour.
  */
private object Builtins {

  private def fail(message: String): Nothing = throw new TemplateException(
    message
  )

  def attribute(value: Value, name: String): Option[Value] = value match {
    case Undefined(missing) => fail(s"'$missing' is undefined")
    case Namespace(fields)  => Some(fields.getOrElse(name, Undefined(name)))
    case Text(s)            => stringMethod(s, name)
    case Dict(entries)      => dictMethod(entries, name)
    case _                  => None
  }

  def item(value: Value, key: Value): Value = (value, key) match {
    case (Undefined(missing), _) => fail(s"'$missing' is undefined")
    case (Dict(entries), k)      => entries.getOrElse(k.show, Undefined(k.show))
    case (Namespace(fields), k)  => fields.getOrElse(k.show, Undefined(k.show))
    case (Sequence(values), Integer(i)) =>
      val at = if (i < 0) values.size + i else i
      if (at >= 0 && at < values.size) values(at.toInt) else Undefined(s"[$i]")
    case (Text(s), Integer(i)) =>
      val at = if (i < 0) s.length + i else i
      if (at >= 0 && at < s.length) Text(s.charAt(at.toInt).toString)
      else Undefined(s"[$i]")
    case _ => Undefined(key.show)
  }

  def slice(
      value: Value,
      start: Option[Value],
      stop: Option[Value],
      step: Option[Value]
  ): Value = {
    def int(v: Option[Value]): Option[Int] = v.collect { case Integer(n) =>
      n.toInt
    }
    def indices(size: Int): Seq[Int] = {
      val s = int(step).getOrElse(1)
      if (s == 0) fail("slice step cannot be zero")
      def clamp(i: Int, low: Int, high: Int) =
        math.max(low, math.min(high, if (i < 0) i + size else i))
      if (s > 0) {
        val from = int(start).map(clamp(_, 0, size)).getOrElse(0)
        val to = int(stop).map(clamp(_, 0, size)).getOrElse(size)
        from until to by s
      } else {
        val from = int(start).map(clamp(_, -1, size - 1)).getOrElse(size - 1)
        val to = int(stop).map(clamp(_, -1, size - 1)).getOrElse(-1)
        from until to by s
      }
    }
    value match {
      case Items(values) => Items(indices(values.size).map(values).toVector)
      case Tuple(values) => Tuple(indices(values.size).map(values).toVector)
      case Text(s)       => Text(indices(s.length).map(s.charAt).mkString)
      case other => fail(s"'${other.typeName}' object is not subscriptable")
    }
  }

  private def method(name: String)(
      body: (Seq[Value], Map[String, Value]) => Value
  ): Option[Value] =
    Some(Function(name, body))

  private def optionalText(v: Option[Value]): Option[String] = v.collect {
    case Text(s) => s
  }

  private def stringMethod(s: String, name: String): Option[Value] =
    name match {
      case "startswith" =>
        method(name)((a, _) =>
          Bool(a.head match {
            case Sequence(options) => options.exists(o => s.startsWith(o.show))
            case prefix            => s.startsWith(prefix.show)
          })
        )
      case "endswith" =>
        method(name)((a, _) =>
          Bool(a.head match {
            case Sequence(options) => options.exists(o => s.endsWith(o.show))
            case suffix            => s.endsWith(suffix.show)
          })
        )
      case "strip" =>
        method(name)((a, _) =>
          Text(strip(s, optionalText(a.headOption), left = true, right = true))
        )
      case "lstrip" =>
        method(name)((a, _) =>
          Text(strip(s, optionalText(a.headOption), left = true, right = false))
        )
      case "rstrip" =>
        method(name)((a, _) =>
          Text(strip(s, optionalText(a.headOption), left = false, right = true))
        )
      case "upper"      => method(name)((_, _) => Text(s.toUpperCase))
      case "lower"      => method(name)((_, _) => Text(s.toLowerCase))
      case "title"      => method(name)((_, _) => Text(title(s)))
      case "capitalize" =>
        method(name)((_, _) =>
          Text(s.take(1).toUpperCase + s.drop(1).toLowerCase)
        )
      case "replace" =>
        method(name)((a, _) => Text(s.replace(a(0).show, a(1).show)))
      case "split" =>
        method(name) { (a, k) =>
          val separator = optionalText(a.headOption.orElse(k.get("sep")))
          val limit = a
            .lift(1)
            .orElse(k.get("maxsplit"))
            .collect { case Integer(n) => n.toInt }
            .getOrElse(-1)
          Items(split(s, separator, limit).map(Text(_)).toVector)
        }
      case "find" =>
        method(name)((a, _) => Integer(s.indexOf(a.head.show).toLong))
      case "count" =>
        method(name)((a, _) =>
          Integer(s.sliding(a.head.show.length).count(_ == a.head.show).toLong)
        )
      case "join" =>
        method(name)((a, _) => Text(a.head.iterate.map(_.show).mkString(s)))
      case _ => None
    }

  private def dictMethod(
      entries: mutable.LinkedHashMap[String, Value],
      name: String
  ): Option[Value] = name match {
    case "get" =>
      method(name)((a, _) =>
        entries.getOrElse(a.head.show, a.lift(1).getOrElse(NoneValue))
      )
    case "items" =>
      method(name)((_, _) =>
        Items(entries.map((k, v) => Tuple(Vector(Text(k), v))).toVector)
      )
    case "keys" =>
      method(name)((_, _) => Items(entries.keys.map(Text(_)).toVector))
    case "values" => method(name)((_, _) => Items(entries.values.toVector))
    case _        => None
  }

  /** Python's `str.strip` family: whitespace, or the given characters. */
  private def strip(
      s: String,
      characters: Option[String],
      left: Boolean,
      right: Boolean
  ): String = {
    val drop: Char => Boolean =
      characters.fold[Char => Boolean](_.isWhitespace)(cs => cs.contains(_))
    val start = if (left) s.indexWhere(c => !drop(c)) match {
      case -1 => s.length; case i => i
    }
    else 0
    val end = if (right) s.lastIndexWhere(c => !drop(c)) + 1 else s.length
    if (start >= end) "" else s.substring(start, end)
  }

  /** Python's `str.split`: on whitespace runs without a separator. */
  private def split(
      s: String,
      separator: Option[String],
      limit: Int
  ): Seq[String] = separator match {
    case None =>
      val words = s.trim.split("\\s+", if (limit < 0) 0 else limit + 1).toSeq
      if (s.trim.isEmpty) Nil else words
    case Some(sep) =>
      if (sep.isEmpty) fail("empty separator")
      val parts = mutable.ArrayBuffer.empty[String]
      var from = 0
      var at = s.indexOf(sep)
      while (at >= 0 && (limit < 0 || parts.size < limit)) {
        parts += s.substring(from, at)
        from = at + sep.length
        at = s.indexOf(sep, from)
      }
      parts += s.substring(from)
      parts.toSeq
  }

  private def title(s: String): String = {
    val out = new StringBuilder
    var previousLetter = false
    s.foreach { c =>
      out += (if (previousLetter) c.toLower else c.toUpper)
      previousLetter = c.isLetter
    }
    out.toString
  }

  def binary(operator: String, left: Value, right: Value): Value =
    (operator, left, right) match {
      case ("==", l, r)                => Bool(Value.equal(l, r))
      case ("!=", l, r)                => Bool(!Value.equal(l, r))
      case ("<", l, r)                 => Bool(Value.compare(l, r) < 0)
      case (">", l, r)                 => Bool(Value.compare(l, r) > 0)
      case ("<=", l, r)                => Bool(Value.compare(l, r) <= 0)
      case (">=", l, r)                => Bool(Value.compare(l, r) >= 0)
      case ("in", l, r)                => Bool(contains(r, l))
      case ("not in", l, r)            => Bool(!contains(r, l))
      case ("~", l, r)                 => Text(l.show + r.show)
      case ("+", Text(a), Text(b))     => Text(a + b)
      case ("+", Items(a), Items(b))   => Items(a ++ b)
      case ("*", Text(a), Integer(n))  => Text(a * n.toInt)
      case ("*", Items(a), Integer(n)) => Items(Vector.fill(n.toInt)(a).flatten)
      case (op, Integer(a), Integer(b)) =>
        op match {
          case "+"  => Integer(a + b)
          case "-"  => Integer(a - b)
          case "*"  => Integer(a * b)
          case "/"  => Real(a.toDouble / b)
          case "//" => Integer(Math.floorDiv(a, b))
          case "%"  => Integer(Math.floorMod(a, b))
          case "**" => Integer(BigInt(a).pow(b.toInt).toLong)
          case _    => fail(s"unknown operator $op")
        }
      case (op, l, r) =>
        (Value.number(l), Value.number(r), op) match {
          case (Some(a), Some(b), "+")  => Real(a + b)
          case (Some(a), Some(b), "-")  => Real(a - b)
          case (Some(a), Some(b), "*")  => Real(a * b)
          case (Some(a), Some(b), "/")  => Real(a / b)
          case (Some(a), Some(b), "//") => Real(math.floor(a / b))
          case (Some(a), Some(b), "%")  => Real(a - b * math.floor(a / b))
          case (Some(a), Some(b), "**") => Real(math.pow(a, b))
          case _ if op == "+" && l.isInstanceOf[Text] =>
            fail(s"can only concatenate str (not \"${r.typeName}\") to str")
          case _ =>
            fail(
              s"unsupported operand type(s) for $op: '${l.typeName}' and '${r.typeName}'"
            )
        }
    }

  private def contains(container: Value, element: Value): Boolean =
    container match {
      case Text(s)          => s.contains(element.show)
      case Sequence(values) => values.exists(Value.equal(_, element))
      case Dict(entries)    => entries.contains(element.show)
      case Namespace(f)     => f.contains(element.show)
      case Undefined(_)     => false
      case other            =>
        fail(s"argument of type '${other.typeName}' is not iterable")
    }

  def filter(
      name: String,
      value: Value,
      positional: Seq[Value],
      keywords: Map[String, Value]
  ): Value = {
    def argument(i: Int, key: String): Option[Value] =
      positional.lift(i).orElse(keywords.get(key))
    name match {
      case "trim" =>
        Text(
          strip(
            value.show,
            optionalText(argument(0, "chars")),
            left = true,
            right = true
          )
        )
      case "length" | "count" =>
        value match {
          case Text(s)      => Integer(s.codePointCount(0, s.length).toLong)
          case Undefined(_) => Integer(0)
          case other        => Integer(other.iterate.size.toLong)
        }
      case "tojson" =>
        val indent = argument(0, "indent").collect { case Integer(n) =>
          n.toInt
        }
        Text(Value.toJson(value, indent))
      case "default" | "d" =>
        val fallback = argument(0, "default_value").getOrElse(Text(""))
        val boolean = argument(1, "boolean").exists(_.truthy)
        if (value.isUndefined || (boolean && !value.truthy)) fallback else value
      case "upper"      => Text(value.show.toUpperCase)
      case "lower"      => Text(value.show.toLowerCase)
      case "capitalize" =>
        Text(value.show.take(1).toUpperCase + value.show.drop(1).toLowerCase)
      case "title"  => Text(title(value.show))
      case "string" => Text(value.show)
      case "safe"   => value
      case "int"    =>
        value match {
          case Integer(n) => Integer(n)
          case Real(n)    => Integer(n.toLong)
          case Text(s)    =>
            s.trim.toLongOption.map(Integer(_)).getOrElse(Integer(0))
          case _ => Integer(0)
        }
      case "float" => Value.number(value).map(Real(_)).getOrElse(Real(0))
      case "abs"   =>
        value match {
          case Integer(n) => Integer(math.abs(n))
          case Real(n)    => Real(math.abs(n))
          case other => fail(s"bad operand type for abs(): '${other.typeName}'")
        }
      case "join" =>
        val separator = argument(0, "d").map(_.show).getOrElse("")
        Text(value.iterate.map(_.show).mkString(separator))
      case "list"    => Items(value.iterate)
      case "first"   => value.iterate.headOption.getOrElse(Undefined("first"))
      case "last"    => value.iterate.lastOption.getOrElse(Undefined("last"))
      case "replace" =>
        Text(
          value.show.replace(
            argument(0, "old").map(_.show).getOrElse(""),
            argument(1, "new").map(_.show).getOrElse("")
          )
        )
      case "items" =>
        value match {
          case Dict(entries) =>
            Items(entries.map((k, v) => Tuple(Vector(Text(k), v))).toVector)
          case Undefined(_) => Items(Vector.empty)
          case other => fail(s"items: '${other.typeName}' is not a mapping")
        }
      case "dictsort" =>
        value match {
          case Dict(entries) =>
            val caseSensitive = argument(0, "case_sensitive").exists(_.truthy)
            val sorted = entries.toVector.sortBy((k, _) =>
              if (caseSensitive) k else k.toLowerCase
            )
            Items(sorted.map((k, v) => Tuple(Vector(Text(k), v))))
          case other => fail(s"dictsort: '${other.typeName}' is not a mapping")
        }
      case "map" =>
        keywords.get("attribute") match {
          case Some(attribute) =>
            Items(value.iterate.map(item => Builtins.item(item, attribute)))
          case None =>
            val filterName = positional.headOption
              .map(_.show)
              .getOrElse(fail("map needs a filter"))
            Items(
              value.iterate.map(item =>
                filter(filterName, item, positional.drop(1), Map.empty)
              )
            )
        }
      case "selectattr" | "rejectattr" =>
        val attribute = positional.head
        val keep = (item: Value) =>
          positional.lift(1) match {
            case Some(Text(testName)) =>
              test(testName, Builtins.item(item, attribute), positional.drop(2))
            case _ => Builtins.item(item, attribute).truthy
          }
        Items(
          value.iterate.filter(item => keep(item) == (name == "selectattr"))
        )
      case other => fail(s"no filter named '$other'")
    }
  }

  def test(name: String, value: Value, arguments: Seq[Value]): Boolean =
    name match {
      case "defined"   => !value.isUndefined
      case "undefined" => value.isUndefined
      case "none"      => value == NoneValue
      case "string"    => value.isInstanceOf[Text]
      case "number"  => value.isInstanceOf[Integer] || value.isInstanceOf[Real]
      case "integer" => value.isInstanceOf[Integer]
      case "float"   => value.isInstanceOf[Real]
      case "boolean" => value.isInstanceOf[Bool]
      case "true"    => value == Bool(true)
      case "false"   => value == Bool(false)
      case "mapping" =>
        value.isInstanceOf[Dict] || value.isInstanceOf[Namespace]
      case "sequence" =>
        value match {
          case Sequence(_) | Text(_) | Dict(_) => true; case _ => false
        }
      // Jinja's Undefined is iterable (as nothing)
      case "iterable" =>
        value match {
          case Sequence(_) | Text(_) | Dict(_) | Undefined(_) => true;
          case _                                              => false
        }
      case "callable"                         => value.isInstanceOf[Function]
      case "eq" | "equalto" | "==" | "sameas" =>
        Value.equal(value, arguments.head)
      case "ne"    => !Value.equal(value, arguments.head)
      case "in"    => contains(arguments.head, value)
      case "lower" => value.show == value.show.toLowerCase
      case "upper" => value.show == value.show.toUpperCase
      case "even"  =>
        value match { case Integer(n) => n % 2 == 0; case _ => false }
      case "odd" =>
        value match { case Integer(n) => n % 2 != 0; case _ => false }
      case "divisibleby" =>
        (value, arguments.head) match {
          case (Integer(n), Integer(d)) => n % d == 0
          case _                        => false
        }
      case other => fail(s"no test named '$other'")
    }

  /** Python's `strftime` for the codes templates print dates with. */
  private def strftime(format: String, time: LocalDateTime): String = {
    val out = new StringBuilder
    var i = 0
    while (i < format.length) {
      if (format.charAt(i) == '%' && i + 1 < format.length) {
        out ++= (format.charAt(i + 1) match {
          case 'Y' => f"${time.getYear}%04d"
          case 'y' => f"${time.getYear % 100}%02d"
          case 'm' => f"${time.getMonthValue}%02d"
          case 'd' => f"${time.getDayOfMonth}%02d"
          case 'H' => f"${time.getHour}%02d"
          case 'M' => f"${time.getMinute}%02d"
          case 'S' => f"${time.getSecond}%02d"
          case 'B' =>
            time.getMonth.getDisplayName(
              java.time.format.TextStyle.FULL,
              java.util.Locale.ENGLISH
            )
          case 'b' =>
            time.getMonth.getDisplayName(
              java.time.format.TextStyle.SHORT,
              java.util.Locale.ENGLISH
            )
          case 'A' =>
            time.getDayOfWeek.getDisplayName(
              java.time.format.TextStyle.FULL,
              java.util.Locale.ENGLISH
            )
          case 'a' =>
            time.getDayOfWeek.getDisplayName(
              java.time.format.TextStyle.SHORT,
              java.util.Locale.ENGLISH
            )
          case '%' => "%"
          case c   => fail(s"strftime: %$c is not supported")
        })
        i += 2
      } else {
        out += format.charAt(i)
        i += 1
      }
    }
    out.toString
  }

  def globals(now: () => LocalDateTime): Map[String, Value] = Map(
    "range" -> Function(
      "range",
      (a, _) => {
        val numbers = a.map {
          case Integer(n) => n
          case other => fail(s"range: '${other.typeName}' is not an integer")
        }
        val (start, stop, step) = numbers match {
          case Seq(stop)              => (0L, stop, 1L)
          case Seq(start, stop)       => (start, stop, 1L)
          case Seq(start, stop, step) => (start, stop, step)
          case _                      => fail("range takes 1 to 3 arguments")
        }
        Items(Range.Long(start, stop, step).map(Integer(_)).toVector)
      }
    ),
    "namespace" -> Function(
      "namespace",
      (_, k) => Namespace(mutable.LinkedHashMap.from(k))
    ),
    "raise_exception" -> Function(
      "raise_exception",
      (a, _) => fail(a.headOption.map(_.show).getOrElse("raised"))
    ),
    "strftime_now" -> Function(
      "strftime_now",
      (a, _) => Text(strftime(a.head.show, now()))
    )
  )
}
