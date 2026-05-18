
using Gridap
using PartitionedArrays
using GridapDistributed

using Gridap.FESpaces, Gridap.Algebra

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

dof_ids = get_free_dof_ids(dist_V)

serial_Ω = Triangulation(serial_model)
serial_dΩ = Measure(serial_Ω,2)

dist_Ω = Triangulation(dist_model)
dist_dΩ = Measure(dist_Ω,2)

dist_Ωg = Triangulation(GridapDistributed.with_ghost,dist_model)
dist_dΩg = Measure(dist_Ωg,2)

serial_a(u,v) = ∫(u⋅v)*serial_dΩ
dist_a(u,v) = ∫(u⋅v)*dist_dΩ
dist_ag(u,v) = ∫(u⋅v)*dist_dΩg

serial_A = assemble_matrix(serial_a,serial_V,serial_V)

assem = SparseMatrixAssembler(dist_V,dist_V,GridapDistributed.Assembled())
dist_A_AS = assemble_matrix(dist_a,assem,dist_V,dist_V)

assem = SparseMatrixAssembler(dist_V,dist_V,GridapDistributed.LocallyAssembled())
dist_A_LA = assemble_matrix(dist_ag,assem,dist_V,dist_V)

assem = SparseMatrixAssembler(dist_V,dist_V,GridapDistributed.SubAssembled())
dist_A_SA = assemble_matrix(dist_a,assem,dist_V,dist_V)

all(centralize(dist_A_AS) - serial_A .< 1e-10)

x_AS = prand(partition(axes(dist_A_AS,2)))
x_LA = GridapDistributed.change_ghost(x_AS,axes(dist_A_LA,2))
x_SA = GridapDistributed.change_ghost(x_AS,axes(dist_A_SA,2))

norm(dist_A_AS*x_AS - dist_A_LA*x_LA)
norm(dist_A_AS*x_AS - dist_A_SA*x_SA)

assemble_matrix!(dist_a,dist_A_AS,assem,dist_V,dist_V)
norm(dist_A_AS*x_AS - dist_A_SA*x_SA)

############################################################################################
