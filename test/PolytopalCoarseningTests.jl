module PolytopalCoarseningTests

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

np = (2,2)
ranks = with_mpi() do distribute
  distribute(LinearIndices((prod(np),)))
end

Dc = 2
fmodel = distributed_voronoi(ranks,np,(8,8),(0,1,0,1))
writevtk(fmodel,"tmp/fmodel")

fgids = partition(get_cell_gids(fmodel))
patch_cells = map(fgids) do fids
  Table([collect(own_to_local(fids))])
end
ptopo = Geometry.PatchTopology(get_grid_topology(fmodel), patch_cells)

cmodel, glues = Adaptivity.coarsen(fmodel,ptopo; return_glue=true)
writevtk(cmodel, "tmp/cmodel")

end # module