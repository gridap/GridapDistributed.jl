
using Gridap
using PartitionedArrays
using GridapDistributed

np = (1,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

nc = (2,4)
serial_model = CartesianDiscreteModel((0,1,0,1),nc)
dist_model = CartesianDiscreteModel(ranks,np,(0,1,0,1),nc)

cids = get_cell_gids(dist_model)

reffe = ReferenceFE(lagrangian,Float64,1)
serial_V = TestFESpace(serial_model,reffe)
dist_V = TestFESpace(dist_model,reffe)

serial_ids = get_free_dof_ids(serial_V)
dist_ids = get_free_dof_ids(dist_V)

serial_Ω = Triangulation(serial_model)
serial_dΩ = Measure(serial_Ω,2)

dist_Ω = Triangulation(dist_model)
dist_dΩ = Measure(dist_Ω,2)

serial_a1(u,v) = ∫(u⋅v)*serial_dΩ
serial_A1 = assemble_matrix(serial_a1,serial_V,serial_V)

dist_a1(u,v) = ∫(u⋅v)*dist_dΩ
dist_A1 = assemble_matrix(dist_a1,dist_V,dist_V)
all(centralize(dist_A1) - serial_A1 .< 1e-10)

