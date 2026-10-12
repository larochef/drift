package drift.runner.ops

import drift.runner.tensor.*

import java.lang.foreign.ValueLayout.*

/** The reference backend's normalisations and activations.
  */
private[ops] trait CpuNormalisation extends CpuBase {

  private def activate(kind: Activation, v: Double): Double = kind match {
    case Activation.Silu     => v / (1 + math.exp(-v))
    case Activation.GeluTanh =>
      0.5 * v * (1 + math.tanh(
        math.sqrt(2 / math.Pi) * (v + 0.044715 * v * v * v)
      ))
    case Activation.GeluErf => 0.5 * v * (1 + CpuOps.erf(v / math.sqrt(2)))
    case Activation.Sigmoid => 1 / (1 + math.exp(-v))
  }

  def scaledActivation(
      kind: Activation,
      inputScale: Float,
      x: Tensor,
      out: Tensor
  ): Unit = {
    Ops.checkElementwise("activation", x, out)
    val input = reader(x)
    fill(out)(i => activate(kind, inputScale * input(i)))
  }

  def gated(kind: Activation, gate: Tensor, up: Tensor, out: Tensor): Unit = {
    Ops.checkElementwise("gated", gate, up, out)
    val (g, u) = (reader(gate), reader(up))
    fill(out)(i => activate(kind, g(i)) * u(i))
  }

  /** Runs `body(start, columns)` for each row. */
  private def rows(x: Tensor, count: Long)(body: (Long, Long) => Unit): Unit = {
    val columns = x.shape.last
    var row = 0L
    while (row < count) {
      body(row * columns, columns)
      row += 1
    }
  }

  def rmsNorm(
      x: Tensor,
      weight: Tensor,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit = {
    val count = Ops.checkRowWise("rmsNorm", x, out, weight)
    val (input, scale, target) = (reader(x), reader(weight), segment(out))
    rows(x, count) { (start, columns) =>
      val squares =
        (0L until columns).map(c => input(start + c) * input(start + c)).sum
      val factor = 1 / math.sqrt(squares / columns + epsilon)
      (0L until columns).foreach { c =>
        val value = input(start + c) * factor * (scale(c) + weightOffset)
        target.setAtIndex(JAVA_FLOAT, start + c, value.toFloat)
      }
    }
  }

  def groupRmsNorm(
      x: Tensor,
      weight: Tensor,
      groups: Int,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit = {
    val count = Ops.checkRowWise("groupRmsNorm", x, out, weight)
    require(x.shape.last % groups == 0, s"${x.shape} in $groups groups")
    val (input, scale, target) = (reader(x), reader(weight), segment(out))
    val width = x.shape.last / groups
    rows(x, count) { (start, columns) =>
      (0 until groups).foreach { g =>
        val first = start + g * width
        val squares =
          (0L until width).map(c => input(first + c) * input(first + c)).sum
        val factor = 1 / math.sqrt(squares / width + epsilon)
        (0L until width).foreach { c =>
          val value =
            input(first + c) * factor * (scale(g * width + c) + weightOffset)
          target.setAtIndex(JAVA_FLOAT, first + c, value.toFloat)
        }
      }
    }
  }

  def groupNorm(
      x: Tensor,
      groups: Int,
      weight: Tensor,
      bias: Tensor,
      epsilon: Float,
      out: Tensor
  ): Unit = {
    val (rows, channels) = Ops.checkGroupNorm(x, groups, weight, bias, out)
    val (input, scale, shift) = (reader(x), reader(weight), reader(bias))
    val width = channels / groups
    val statistics = (0 until groups).map { g =>
      def values =
        (0L until rows).iterator.flatMap(r =>
          (0 until width).iterator.map(c => input(r * channels + g * width + c))
        )
      val count = rows * width
      val mean = values.sum / count
      val variance = values.map(v => (v - mean) * (v - mean)).sum / count
      (mean, 1 / math.sqrt(variance + epsilon))
    }
    fill(out) { index =>
      val c = (index % channels).toInt
      val (mean, factor) = statistics(c / width)
      (input(index) - mean) * factor * scale(c) + shift(c)
    }
  }

  def layerNorm(
      x: Tensor,
      weight: Option[Tensor],
      bias: Option[Tensor],
      epsilon: Float,
      out: Tensor
  ): Unit = {
    val count = Ops.checkRowWise("layerNorm", x, out, (weight ++ bias).toSeq*)
    val (input, target) = (reader(x), segment(out))
    val (scale, shift) = (weight.map(reader), bias.map(reader))
    rows(x, count) { (start, columns) =>
      val mean = (0L until columns).map(c => input(start + c)).sum / columns
      val variance =
        (0L until columns)
          .map(c => math.pow(input(start + c) - mean, 2))
          .sum / columns
      val factor = 1 / math.sqrt(variance + epsilon)
      (0L until columns).foreach { c =>
        val normal = (input(start + c) - mean) * factor
        val value = normal * scale.fold(1.0)(_(c)) + shift.fold(0.0)(_(c))
        target.setAtIndex(JAVA_FLOAT, start + c, value.toFloat)
      }
    }
  }

  def softmax(x: Tensor, scale: Float, out: Tensor): Unit = {
    val count = Ops.checkRowWise("softmax", x, out)
    val (input, target) = (reader(x), segment(out))
    rows(x, count) { (start, columns) =>
      val scaled = (0L until columns).map(c => input(start + c) * scale)
      val largest = scaled.max
      val exponentials = scaled.map(v => math.exp(v - largest))
      val sum = exponentials.sum
      exponentials.zipWithIndex.foreach { (value, c) =>
        target.setAtIndex(JAVA_FLOAT, start + c, (value / sum).toFloat)
      }
    }
  }

  def gatedRmsNorm(
      x: Tensor,
      weight: Tensor,
      gate: Tensor,
      gateActivation: Activation,
      epsilon: Float,
      out: Tensor
  ): Unit = {
    Ops.requireSameShape("gatedRmsNorm", x, gate)
    rmsNorm(x, weight, epsilon, 0f, out)
    gated(gateActivation, gate, out, out)
  }
}
