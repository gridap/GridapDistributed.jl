module ConstantFESpacesTests

using Test
using Gridap
using GridapDistributed, PartitionedArrays

using Gridap.FESpaces, Gridap.Arrays, Gridap.Algebra

function main(distribute,np)
  @assert prod(np) == 4
  ranks = distribute(LinearIndices((prod(np),)))

  model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(10,10))

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2)

  reffe = ReferenceFE(lagrangian, Float64, 1)
  U = FESpace(model,reffe)
  V = ConstantFESpace(model)
  X = MultiFieldFESpace([U,V])

  gids = get_free_dof_ids(V)
  @test length(gids) == 1

  a((u,λ),(v,μ)) = ∫(u*v + λ*v + μ*u)dΩ 
  A = assemble_matrix(a,X,X)

  V2 = ConstantFESpace(model;constraint_type=:local)
  X2 = MultiFieldFESpace([U,V2])

  gids = get_free_dof_ids(V2)
  @test length(gids) == 4

  A2 = assemble_matrix(a,X2,X2)
end

end
