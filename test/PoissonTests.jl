module PoissonTests
using SparseMatricesCSR
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using Test

function main(distribute, parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks, parts,domain,cells)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet",[1,2,3,5,7])
  add_tag_from_tags!(labels,"neumann",[4,6,8])

  Ω = Triangulation(model)
  Γn = Boundary(model,tags="neumann")
  n_Γn = get_normal_vector(Γn)

  k = 2
  u((x,y)) = (x+y)^k
  f(x) = -Δ(u,x)
  g = n_Γn⋅∇(u)

  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="dirichlet")
  U = TrialFESpace(u,V)

  dΩ = Measure(Ω,2*k)
  dΓn = Measure(Γn,2*k)

  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  l(v) = ∫( v*f )dΩ + ∫( v*g )dΓn
  assem=SparseMatrixAssembler(SparseMatrixCSR{0,Float64,Int},Vector{Float64},U,V)
  op = AffineFEOperator(a,l,U,V,assem)

  uh = solve(op)
  eh = u - uh
  @test sqrt(sum( ∫(abs2(eh))dΩ )) < 1.0e-9

  # Now with DG
  h = 2
  γ = 10

  V_dg = FESpace(model,reffe,conformity=:L2)

  Λ = Skeleton(model)
  Γd = Boundary(model,tags="dirichlet")

  dΛ = Measure(Λ,2*k)
  dΓd = Measure(Γd,2*k)

  n_Γd = get_normal_vector(Γd)
  n_Λ = get_normal_vector(Λ)

  a_dg(u,v) =
    ∫( ∇(v)⋅∇(u) )*dΩ +
    ∫( (γ/h)*v*u  - v*(n_Γd⋅∇(u)) - (n_Γd⋅∇(v))*u )*dΓd +
    ∫( (γ/h)*jump(v*n_Λ)⋅jump(u*n_Λ) -
       jump(v*n_Λ)⋅mean(∇(u)) -
       mean(∇(v))⋅jump(u*n_Λ) )*dΛ

  l_dg(v) =
    ∫( v*f )*dΩ +
    ∫( v*g )dΓn +
    ∫( (γ/h)*v*u - (n_Γd⋅∇(v))*u )*dΓd

  op = AffineFEOperator(a_dg,l_dg,V_dg,V_dg)
  uh = solve(op)
  eh = u - uh
  @test sqrt(sum( ∫(abs2(eh))dΩ )) < 1.0e-9

end

end # module
