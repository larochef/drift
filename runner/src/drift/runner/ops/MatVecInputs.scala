package drift.runner.ops

/** What quantized weights meet in a matrix-vector product: `Int8` quantizes x
  * per 32 values and multiplies four codes per `v_dot4`, as llama.cpp does;
  * `Float` keeps x as is and decodes the weights to floats in registers.
  */
enum MatVecInputs {
  case Int8
  case Float
}
