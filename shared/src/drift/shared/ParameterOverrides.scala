package drift.shared

/** What a layer is worth storing (`specs/16-parameter-resolution.md`).
  *
  * An override that repeats the value a lower layer already gives changes
  * nothing on a command line, and a removal of a flag nothing below sets
  * removes nothing — yet both read like a decision in the form and in the file.
  * One of them is how a flag meant to be *removed* came to be stored as an
  * override of the value it already had, invisibly, because the number was the
  * same either way (François, 2026-09-20).
  *
  * So a record carries only the difference; the effective list stays computed
  * (`CommandLine.resolveParameters`). This is the one implementation of that
  * rule: the backend prunes what it stores, and nothing else has to remember
  * to.
  */
object ParameterOverrides {

  /** One layer's parameters and removals, with everything the layers below
    * already decide dropped.
    *
    * Removals are applied to the layer's own values first, because that is what
    * resolution does: a flag a layer both sets and removes is never passed, so
    * neither half is worth keeping unless the removal still has something below
    * it to remove.
    */
  def prunedParameters(
      parameters: Map[String, String],
      removed: List[String],
      inherited: List[ResolvedParameter]
  ): (Map[String, String], List[String]) = {
    val below =
      inherited.map(parameter => parameter.flag -> parameter.value).toMap
    val removals = removed.distinct
    val own = parameters.filterNot((flag, _) => removals.contains(flag))
    (
      own.filterNot((flag, value) => below.get(flag).contains(value)),
      removals.filter(below.contains)
    )
  }

  /** A run configuration as it should be stored: pruned against its
    * architecture and its assigned models, and deliberately *not* against any
    * runtime. A runtime is chosen at launch and changes under a configuration
    * that was never edited, so an override matching one build's default is
    * still a decision about every other build.
    */
  def prunedConfiguration(
      configuration: RunConfiguration,
      architecture: Option[Architecture],
      models: List[Model]
  ): RunConfiguration =
    architecture match {
      // Nothing to compare against: an unknown architecture keeps what was
      // typed rather than losing it.
      case None        => configuration
      case Some(shape) =>
        val (parameters, removed) = prunedParameters(
          configuration.overriddenParameters,
          configuration.removedParameters,
          CommandLine.inheritedParameters(configuration, shape, models)
        )
        configuration.copy(
          overriddenParameters = parameters,
          removedParameters = removed
        )
    }

  /** A model as it should be stored.
    *
    * Pruned against every architecture that can assign it, not against the one
    * whose card it was edited from: a family's models are shared
    * (`specs/02-model-registry.md`), so a value that repeats one architecture's
    * default may be the only thing keeping another's off the command line. A
    * value is redundant when every architecture that could assign the model
    * already gives that flag that value; a removal is worth keeping as soon as
    * one of them sets the flag at all.
    */
  def prunedModel(model: Model, architectures: List[Architecture]): Model = {
    val assignable = architectures.filter(
      _.checkpoints.exists(_.familyId == model.familyId)
    )
    // An orphan model — no architecture names its family — has no layer below
    // it to compare against, so nothing of it is redundant.
    if (assignable.isEmpty) model
    else {
      val removals = model.removedParameters.distinct
      val own = model.parameters.filterNot((flag, _) => removals.contains(flag))
      model.copy(
        parameters = own.filterNot((flag, value) =>
          assignable.forall(_.defaultParameters.get(flag).contains(value))
        ),
        removedParameters = removals.filter(flag =>
          assignable.exists(_.defaultParameters.contains(flag))
        )
      )
    }
  }
}
