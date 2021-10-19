module PLaplacianTests

using Gridap
using Gridap.Algebra
using GridapDistributed
using PartitionedArrays
using Test

function main(parts)

  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(parts,domain,cells)
  #model = CartesianDiscreteModel(domain,cells)

  k = 1
  u((x,y)) = (x+y)^k
  σ(∇u) =(1.0+∇u⋅∇u)*∇u
  dσ(∇du,∇u) = 2*∇u⋅∇du + (1.0+∇u⋅∇u)*∇du
  f(x) = -divergence(y->σ(∇(u,y)),x)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*k)
  r(u,v) = ∫( ∇(v)⋅(σ∘∇(u)) - v*f )dΩ
  j(u,du,v) = ∫( ∇(v)⋅(dσ∘(∇(du),∇(u))) )dΩ

  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)

  op = FEOperator(r,j,U,V)

  uh = zero(U)
  b,A = residual_and_jacobian(op,uh)
  _A = copy(A)
  _b = copy(b)
  residual_and_jacobian!(_b,_A,op,uh)
  @test (norm(_b-b)+1) ≈ 1
  x = similar(b,Float64,axes(A,2))
  fill!(x,1)
  @test (norm(A*x-_A*x)+1) ≈ 1

  nls = NLSolver(show_trace=i_am_main(parts), method=:newton)
  solver = FESolver(nls)
  uh = solve(solver,op)

  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩ)) < 1.0e-9

end

end # module
