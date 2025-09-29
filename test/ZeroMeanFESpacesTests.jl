module ZeroMeanFESpacesTests

using Gridap
using Gridap.FESpaces, Gridap.MultiField, Gridap.Algebra
using GridapDistributed
using PartitionedArrays
using Test

function main(distribute, parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(4,4))

  Ω = Triangulation(model)
  dΩ = Measure(Ω,4)

  u_ex(x) = VectorValue(x[2],-x[1])
  p_ex(x) = x[1] + 2*x[2]
  p_mean = sum(∫(p_ex)*dΩ)

  order = 2
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

  V  = FESpace(model,reffe_u;dirichlet_tags="boundary")
  U  = TrialFESpace(V,u_ex)
  Q  = FESpace(model,reffe_p;conformity=:L2)
  Q0 = FESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)
  @test isa(Q0,GridapDistributed.DistributedZeroMeanFESpace)

  X = MultiFieldFESpace([U,Q0])
  Y = MultiFieldFESpace([V,Q0])

  ph_i = interpolate(p_ex, Q0)
  @test abs(sum(∫(ph_i)*dΩ)) < 1.0e-10
  @test abs(sum(∫(ph_i - p_ex + p_mean)*dΩ)) < 1.0e-10

  # Stokes

  a((u,p),(v,q)) = ∫(∇(u)⊙∇(v) - (∇⋅v)*p - q*(∇⋅u))dΩ
  l((v,q)) = a((u_ex,p_ex),(v,q))

  op = AffineFEOperator(a,l,X,Y)
  uh, ph = solve(op);

  l2_error(u,v) = sqrt(sum(∫((u-v)⋅(u-v))*dΩ))
  @test l2_error(uh,u_ex) < 1.0e-10
  @test abs(sum(∫(ph)*dΩ)) < 1.0e-10
  @test l2_error(ph,ph_i) < 1.0e-10

  b(u,q) = ∫(q*(∇⋅u))dΩ
  B = assemble_vector(q -> b(uh,q),Q)
  B0 = assemble_vector(q -> b(uh,q),Q0)
  @test abs(sum(B)) < 1.0e-10
  @test abs(sum(B0)) < 1.0e-10

  # Navier-Stokes

  uh_ex = interpolate(u_ex, U)
  @test l2_error(uh,uh_ex) < 1.0e-10
  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dc(u,du,dv) = ∫(dv⊙(dconv∘(du,∇(du),u,∇(u))))dΩ
  jac((u,p),(du,dp),(dv,dq)) = a((du,dp),(dv,dq)) + dc(u,du,dv)
  res((u,p),(dv,dq)) = a((u,p),(dv,dq)) + c(u,dv) - a((u_ex,p_ex),(dv,dq)) - c(uh_ex,dv)

  op = FEOperator(res,jac,X,Y)
  uh, ph = solve(op);

  @test l2_error(uh,u_ex) < 1.0e-10
  @test abs(sum(∫(ph)*dΩ)) < 1.0e-10
  @test l2_error(ph,ph_i) < 1.0e-10

  B = assemble_vector(q -> b(uh,q),Q)
  B0 = assemble_vector(q -> b(uh,q),Q0)
  @test abs(sum(B)) < 1.0e-10
  @test abs(sum(B0)) < 1.0e-10

  return true
end

end