module MultiFieldTests

using Gridap
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using Test

function main(distribute, parts)
  ranks  = distribute(LinearIndices((prod(parts),)))
  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,cells)
  Ω = Triangulation(model)

  k = 2
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_p = ReferenceFE(lagrangian,Float64,k-1,space=:P)

  u((x,y)) = VectorValue((x+y)^2,(x-y)^2)
  p((x,y)) = x+y
  f(x) = - Δ(u,x) + ∇(p,x)
  g(x) = tr(∇(u,x))

  V = TestFESpace(model,reffe_u,dirichlet_tags="boundary")
  Q = TestFESpace(model,reffe_p,constraint=:zeromean)
  U = TrialFESpace(V,u)
  P = TrialFESpace(Q,p)

  VxQ = MultiFieldFESpace([V,Q])
  UxP = MultiFieldFESpace([U,P]) # This generates again the global numbering
  UxP = TrialFESpace([u,p],VxQ) # This reuses the one computed
  @test length(UxP) == 2

  uh, ph = interpolate([u,p],UxP)
  eu = u - uh
  ep = p - ph

  dΩ = Measure(Ω,2*k)
  @test sqrt(sum(∫( eu⋅eu )dΩ)) < 1.0e-9
  @test sqrt(sum(∫( eu⋅eu )dΩ)) < 1.0e-9

  a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p )*dΩ
  l((v,q)) = ∫( v⋅f - q*g )*dΩ

  op = AffineFEOperator(a,l,UxP,VxQ)
  solver = LinearFESolver(BackslashSolver())
  uh, ph = solve(solver,op)

  eu = u - uh
  ep = p - ph

  writevtk(Ω,"Ω",nsubcells=10,cellfields=["uh"=>uh,"ph"=>ph])

  @test sqrt(sum(∫( eu⋅eu )dΩ)) < 1.0e-9
  @test sqrt(sum(∫( eu⋅eu )dΩ)) < 1.0e-9

end

end # module
