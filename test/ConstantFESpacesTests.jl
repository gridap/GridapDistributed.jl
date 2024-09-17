
using Gridap
using GridapDistributed, PartitionedArrays

using Gridap.FESpaces

np = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(4,4))

Ω = Triangulation(model)
dΩ = Measure(Ω,2)

V = ConstantFESpace(model)

gids = get_free_dof_ids(V)
local_to_global(gids)
local_to_owner(gids)
ghost_to_local(gids)
own_to_local(gids)

uh = zero(V)
sum(∫(uh)dΩ)

