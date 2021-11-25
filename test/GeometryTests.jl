module GeometryTests

using Gridap
using GridapDistributed
using PartitionedArrays
using LinearAlgebra
using Test

function main(parts)

  output = mkpath(joinpath(@__DIR__,"output"))

  if length(size(parts)) == 2
    domain = (0,4,0,4)
    cells = (4,4)
  elseif length(size(parts)) == 3
    domain = (0,4,0,4,0,4)
    cells = (4,4,4)
  end

  model = CartesianDiscreteModel(parts,domain,cells)
  writevtk(model,joinpath(output,"model"))

  @test num_cells(model)==prod(cells)

  gmodel = CartesianDiscreteModel(domain,cells)

  if length(cells) == 2 && prod(cells) == 16
    smodel = simplexify(gmodel)
    if length(parts) == 4
      cell_to_part = [
        1,1,1,1,1,1,1,1,
        1,2,2,2,2,2,2,3,
        3,3,3,3,3,3,3,3,
        3,3,4,4,4,4,4,4]
    else
      cell_to_part = fill(1,num_cells(smodel))
    end
    cell_graph = GridapDistributed.compute_cell_graph(smodel)
    @test LinearAlgebra.issymmetric(cell_graph)
    @test LinearAlgebra.ishermitian(cell_graph)
    dmodel = DiscreteModel(parts,smodel,cell_to_part)
    writevtk(dmodel,joinpath(output,"dmodel"))
  end

  map_parts(model.models,model.gids.partition) do lmodel,gids
    @test test_local_part_face_labelings_consistency(lmodel,gids,gmodel)
  end

  grid = get_grid(model)
  labels = get_grid(model)

  Ω = Triangulation(with_ghost,model)
  writevtk(Ω,joinpath(output,"Ω"))

  Ω = Triangulation(no_ghost,model)
  writevtk(Ω,joinpath(output,"Ω"))

  Γ = Boundary(with_ghost,model,tags="boundary")
  writevtk(Γ,joinpath(output,"Γ"))

  Γ = Boundary(no_ghost,model,tags="boundary")
  writevtk(Γ,joinpath(output,"Γ"))

end

function test_local_part_face_labelings_consistency(lmodel::CartesianDiscreteModel{D},gids,gmodel) where {D}
   local_topology         = lmodel.grid_topology
   global_topology        = gmodel.grid_topology
   local_labelings        = lmodel.face_labeling
   global_labelings       = gmodel.face_labeling
   l_d_to_dface_to_entity = local_labelings.d_to_dface_to_entity
   g_d_to_dface_to_entity = global_labelings.d_to_dface_to_entity
   #traverse local cells
   for cell_lid=1:num_cells(lmodel)
        cell_gid=gids.lid_to_gid[cell_lid]
        for d=0:D-1
             local_cell_to_faces = local_topology.n_m_to_nface_to_mfaces[D+1,d+1]
             global_cell_to_faces = global_topology.n_m_to_nface_to_mfaces[D+1,d+1]
             la = local_cell_to_faces.ptrs[cell_lid]
             lb = local_cell_to_faces.ptrs[cell_lid+1]
             ga = global_cell_to_faces.ptrs[cell_gid]
             gb = global_cell_to_faces.ptrs[cell_gid+1]
             @assert (lb-la)==(gb-ga)
             for i=0:lb-la-1
                 face_lid = local_cell_to_faces.data[la+i]
                 face_gid = global_cell_to_faces.data[ga+i]
                 local_entity = l_d_to_dface_to_entity[d+1][face_lid]
                 global_entity = g_d_to_dface_to_entity[d+1][face_gid]
                 if (local_entity != global_entity)
                     return false
                 end
             end
        end
   end
   return true
end

end # module
