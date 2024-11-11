module AutodiffTests

using Test
using Gridap, Gridap.Algebra
using GridapDistributed
using PartitionedArrays

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,cells)

  u((x,y)) = (x+y)^k
  σ(∇u) = (1.0+∇u⋅∇u)*∇u
  dσ(∇du,∇u) = (2*∇u⋅∇du)*∇u + (1.0+∇u⋅∇u)*∇du
  f(x) = -divergence(y->σ(∇(u,y)),x)

  k = 1
  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*k)
  r(u,v) = ∫( ∇(v)⋅(σ∘∇(u)) - v*f )dΩ
  j(u,du,v) = ∫( ∇(v)⋅(dσ∘(∇(du),∇(u))) )dΩ

  op = FEOperator(r,j,U,V)
  op_AD = FEOperator(r,U,V)

  uh = interpolate(1.0,U)
  A = jacobian(op,uh)
  A_AD = jacobian(op_AD,uh)
  @test reduce(&,map(≈,partition(A),partition(A_AD)))

  g(v) = ∫(0.5*v⋅v)dΩ
  dg(v) = ∫(uh⋅v)dΩ
  b = assemble_vector(dg,U)
  b_AD = assemble_vector(gradient(g,uh),U)
  @test b ≈ b_AD
end

end