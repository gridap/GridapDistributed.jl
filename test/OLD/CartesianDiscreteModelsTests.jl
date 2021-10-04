module CartesianDiscreteModelsTests

using Gridap
using GridapDistributed
using Test

function test_face_labelings_consistency(comm,subdomains,D::Int)
    domain = Int[]
    for i = 1:D
        push!(domain, 0)
        push!(domain, 1)
    end
    domain = Tuple(domain)
    cells = Gridap.Geometry.tfill(4, Val{D}())
    dmodel = CartesianDiscreteModel(comm, subdomains, domain, cells)
    gmodel = CartesianDiscreteModel(domain, cells)

    result = DistributedData{Bool}(
        comm,
        dmodel.models,
        dmodel.gids,
    ) do part, model, gid
        test_local_part_face_labelings_consistency(model, gid, gmodel)
    end
    all(gather(result))
end

function test_local_part_face_labelings_consistency(model::CartesianDiscreteModel{D},gid,gmodel) where {D}
   local_topology         = model.grid_topology
   global_topology        = gmodel.grid_topology
   local_labelings        = model.face_labeling
   global_labelings       = gmodel.face_labeling
   l_d_to_dface_to_entity = local_labelings.d_to_dface_to_entity
   g_d_to_dface_to_entity = global_labelings.d_to_dface_to_entity
   #traverse local cells
   for cell_lid=1:num_cells(model)
        cell_gid=gid.lid_to_gid[cell_lid]
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

function generate_subdomains(D::Int)
    subdomains = Gridap.Geometry.tfill(2, Val{D}())
end 

subdomains = (2,3)
SequentialCommunicator(subdomains) do comm
  domain = (0,1,0,1)
  cells = (10,10)
  model = CartesianDiscreteModel(comm,subdomains,domain,cells)
  writevtk(model,"model")
end 

for D in (1,2,3)
  subdomains = generate_subdomains(D)
  SequentialCommunicator(subdomains) do comm
    @test test_face_labelings_consistency(comm,subdomains,D)
  end
end

end # module
