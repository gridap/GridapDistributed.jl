module PeriodicBCsTests

using Gridap
using GridapDistributed
using PartitionedArrays
using Test

function main(distribute,parts)
  ranks  = distribute(LinearIndices((prod(parts),)))
  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,2π)
  cells = (20,20)
  isperiodic = (false,true)
  model = CartesianDiscreteModel(ranks,parts,domain,cells,isperiodic=isperiodic)

  u((x,y)) = sin(y+π/6)*x
  f(x) = -Δ(u,x)
  k = 2

  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)

  Ω = Interior(model)
  dΩ = Measure(Ω,2*k)

  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  l(v) = ∫( v*f )dΩ
  op = AffineFEOperator(a,l,U,V)

  uh = solve(op)
  eh = u - uh
  @test sqrt(sum( ∫(abs2(eh))dΩ )) < 0.00122

end

end # module
