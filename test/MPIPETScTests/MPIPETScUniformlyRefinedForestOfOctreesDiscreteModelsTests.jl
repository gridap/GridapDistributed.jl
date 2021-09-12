module MPIPETScUniformlyRefinedForestOfOctreesDiscreteModelsTests
  using Gridap
  using GridapDistributed
  using Test

  function run(comm)
    coarse_discrete_model = CartesianDiscreteModel((0,1,0,1),(1,1))
    model=UniformlyRefinedForestOfOctreesDiscreteModel(comm,coarse_discrete_model,3)
    do_on_parts(model) do part, (model,gids)
      print(part, " ", num_cells(model), "\n")
    end
  end

  MPIPETScCommunicator() do comm
    run(comm)
  end

end # module
