module VisualizationTests

using Gridap, GridapDistributed, PartitionedArrays

function half_empty_trian(ranks,model)
  cell_ids = get_cell_gids(model)
  trians = map(ranks,local_views(model),partition(cell_ids)) do rank, model, ids
    cell_mask = zeros(Bool, num_cells(model))
    if rank ∈ (3,4)
      cell_mask[own_to_local(ids)] .= true
    end
    Triangulation(model,cell_mask)
  end
  GridapDistributed.DistributedTriangulation(trians,model)
end

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(8,8))
  V = FESpace(model, ReferenceFE(lagrangian,Float64,1))
  uh = interpolate(x -> x[1]+x[2], V)

  t1 = Triangulation(model)
  writevtk(t1,"output/t1",cellfields=["uh" => uh]) 

  t2 = half_empty_trian(ranks,model)
  writevtk(t2,"output/t2",cellfields=["uh" => uh])
end

end # module