module MacroDiscreteModelsTests

using Gridap
using GridapDistributed
using PartitionedArrays
using Test

function main(distribute,parts;vtk=false)
  ranks = distribute(LinearIndices((prod(parts),)))
  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(8,8))

  macro_model = GridapDistributed.MacroDiscreteModel(model)
  @test num_cells(macro_model) == length(ranks)

  for d in 0:2
    gids = GridapDistributed.get_face_gids(macro_model,d)
  end

  local_labels = GridapDistributed.get_local_face_labeling(macro_model)
  global_labels = GridapDistributed.get_global_face_labeling(macro_model)

  if vtk
    outdir = "tmp/"
    mkpath(outdir)
    GridapDistributed.writevtk_local(macro_model, joinpath(outdir,"macro_model_local"))
    GridapDistributed.writevtk_global(macro_model, joinpath(outdir,"macro_model_global"))
  end

end

end # module