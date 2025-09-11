module PolytopalCoarseningTests

using Test
using Gridap
using GridapDistributed, PartitionedArrays

using Gridap.Adaptivity, Gridap.Geometry, Gridap.Arrays
using Gridap.ReferenceFEs

function distributed_voronoi(ranks,np,nc,domain)
  serial_model = Geometry.voronoi(simplexify(CartesianDiscreteModel(domain,nc)))
  cell_to_rank = zeros(Int,num_cells(serial_model))
  for (r,cells) in enumerate(uniform_partition(1:prod(np),np,nc .+ 1))
    cell_to_rank[cells] .= r
  end
  return DiscreteModel(ranks,serial_model,cell_to_rank)
end

function main(distribute, np)
  ranks = distribute(LinearIndices((prod(np),)))

  fmodel = distributed_voronoi(ranks,np,(8,8),(0,1,0,1))
  #writevtk(fmodel,"tmp/fmodel")

  fgids = partition(get_cell_gids(fmodel))
  patch_cells = map(fgids) do fids
    Table([collect(own_to_local(fids))])
  end
  ptopo = Geometry.PatchTopology(get_grid_topology(fmodel), patch_cells)

  cmodel, glues = Adaptivity.coarsen(fmodel,ptopo; return_glue=true)
  @test GridapDistributed.isconsistent_faces(get_grid_topology(cmodel))
  #writevtk(cmodel, "tmp/cmodel")
end

# ranks = collect(1:2)
# 
# fmodel_serial = Geometry.PolytopalDiscreteModel(CartesianDiscreteModel((0,1,0,1),(4,4)))
# cmodel_serial = Adaptivity.coarsen(
#   fmodel_serial, Geometry.PatchTopology(
#     get_grid_topology(fmodel_serial), Table([[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,15,16]])
#   )
# )
# fmodel_good = DiscreteModel(ranks,fmodel_serial,[1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2])
# cmodel_good = DiscreteModel(ranks,cmodel_serial,[1,2,1,2])
# 
# fmodel = Geometry.PolytopalDiscreteModel(CartesianDiscreteModel(ranks,(2,1),(0,1,0,1),(4,4)))
# cmodel = Adaptivity.coarsen(
#   fmodel, Geometry.PatchTopology(
#     get_grid_topology(fmodel),[Table([[1,2,4,5],[7,8,10,11]]),Table([[2,3,5,6],[8,9,11,12]])]
#   )
# )
# 
# GridapDistributed.isconsistent_faces(get_grid_topology(fmodel))
# GridapDistributed.isconsistent_faces(get_grid_topology(cmodel))


end # module