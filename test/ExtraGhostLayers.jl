
using Gridap
using GridapDistributed
using PartitionedArrays


parts = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(parts),)))
end

nc = (4,4)
model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),nc;ghost=(2,2))

num_cells(model)
map(num_cells,local_views(model))

reffe = ReferenceFE(lagrangian,Float64,1)
V = FESpace(model,reffe)

cell_dof_ids = map(get_cell_dof_ids,local_views(V))

gids = get_free_dof_ids(V)
own_to_local(gids)

Ω = Triangulation(model)
map(num_cells,local_views(Ω))

dΩ = Measure(Ω,2)

f = 1.0
a(u,v) = ∫(∇(u)⋅∇(v))dΩ
l(v) = ∫(f⋅v)dΩ

op = AffineFEOperator(a,l,V,V)



