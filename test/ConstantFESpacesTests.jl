
using Gridap
using GridapDistributed, PartitionedArrays

using Gridap.FESpaces, Gridap.Arrays, Gridap.Algebra

np = (2,2)
ranks = with_mpi() do distribute
  distribute(LinearIndices((prod(np),)))
end

model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(10,10))

Ω = Triangulation(model)
dΩ = Measure(Ω,2)

reffe = ReferenceFE(lagrangian, Float64, 1)
U = FESpace(model,reffe)
V = ConstantFESpace(model)
X = MultiFieldFESpace([U,V])

gids = get_free_dof_ids(V)
local_to_global(gids)
local_to_owner(gids)
ghost_to_local(gids)
own_to_local(gids)

uh = zero(V)
sum(∫(uh)dΩ)

map(ranks) do r
  println("flag 1 from ", r)
end

a((u,λ),(v,μ)) = ∫(u*v + λ*v + μ*u)dΩ 
A = assemble_matrix(a,X,X)

map(ranks) do r
  println("flag 2 from ", r)
end

x = allocate_in_domain(A)

map(ranks) do r
  println("done from ", r)
end
