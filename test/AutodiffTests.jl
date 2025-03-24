module AutodiffTests

using Test
using Gridap, Gridap.Algebra
using GridapDistributed
using PartitionedArrays
using SparseArrays
using ForwardDiff

function main_sf(distribute,parts)
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

  # Skeleton AD
  # I would like to compare the results, but we cannot be using FD in parallel... 
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,2*k)
  g_Λ(v) = ∫(mean(v))*dΛ
  r_Λ(u,v) = ∫(mean(u)*mean(v))*dΛ

  b_Λ_AD = assemble_vector(gradient(g_Λ,uh),U)
  A_Λ_AD = jacobian(FEOperator(r_Λ,U,V),uh)
end

function main_mf(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(4,4))

  k = 2
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_p = ReferenceFE(lagrangian,Float64,k-1;space=:P)

  u(x) = VectorValue(x[2],-x[1])
  V = TestFESpace(model,reffe_u,dirichlet_tags="boundary")
  U = TrialFESpace(V,u)
  Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)

  X = MultiFieldFESpace([U,Q])
  Y = MultiFieldFESpace([V,Q])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+1))
  
  ν = 1.0
  f = VectorValue(0.0,0.0)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dc(u,du,dv) = ∫(dv⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  biform((du,dp),(dv,dq)) = ∫(ν*∇(dv)⊙∇(du) - (∇⋅dv)*dp - (∇⋅du)*dq)dΩ
  liform((dv,dq)) = ∫(dv⋅f)dΩ

  r((u,p),(v,q)) = biform((u,p),(v,q)) + c(u,v) - liform((v,q))
  j((u,p),(du,dp),(dv,dq)) = biform((du,dp),(dv,dq)) + dc(u,du,dv)

  op = FEOperator(r,j,X,Y)
  op_AD = FEOperator(r,X,Y)

  xh = interpolate([VectorValue(1.0,1.0),1.0],X)
  uh, ph = xh
  A = jacobian(op,xh)
  A_AD = jacobian(op_AD,xh)
  @test reduce(&,map(≈,partition(A),partition(A_AD)))

  g((v,q)) = ∫(0.5*v⋅v + 0.5*q*q)dΩ
  dg((v,q)) = ∫(uh⋅v + ph*q)dΩ
  b = assemble_vector(dg,X)
  b_AD = assemble_vector(gradient(g,xh),X)
  @test b ≈ b_AD
end

function main(distribute,parts)
  main_sf(distribute,parts)
  main_mf(distribute,parts)
end

end