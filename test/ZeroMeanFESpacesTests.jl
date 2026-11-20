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
  e_i = ph_i - p_ex + p_mean
  @test abs(sum(∫(ph_i)*dΩ)) < 1.0e-10
  # the L² norm of the error and not the mean value of it, which the shift
  # cancels whatever direction it follows
  @test sqrt(sum(∫(e_i*e_i)*dΩ)) < 1.0e-10

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

  # Moment based bases
  #
  # `modalC0` is not a partition of unity from order 2 on, so the DOF values of
  # 𝔥 are not all ones: the volume is not the sum of the `∫φᵢ`, and the DOF the
  # space fixes cannot be the last one, on which they vanish. Pure Neumann
  # Poisson, whose solution the zero mean value selects, is solved in the same
  # space in the two bases.
  u_N(x) = cos(pi*x[1])
  f_N(x) = pi^2*cos(pi*x[1])
  dΩ_N = Measure(Ω,8)

  uh_N = map((ReferenceFE(lagrangian,Float64,2),ReferenceFE(QUAD,modalC0,Float64,2))) do reffe
    V0 = FESpace(model,reffe;conformity=:H1,constraint=:zeromean)
    @test isa(V0,GridapDistributed.DistributedZeroMeanFESpace)
    # the DOF values of 𝔥 do not vanish on the fixed DOF
    @test V0.metadata.h_diri != 0
    op = AffineFEOperator((u,v) -> ∫( ∇(u)⋅∇(v) )*dΩ_N,
                          v -> ∫( f_N*v )*dΩ_N, TrialFESpace(V0), V0)
    uh = solve(op)
    @test abs(sum(∫(uh)*dΩ_N)) < 1.0e-10
    uh
  end
  e_N = uh_N[1] - uh_N[2]
  @test sqrt(sum(∫(e_N*e_N)*dΩ_N)) < 1.0e-10
  e_ex = uh_N[1] - u_N
  @test sqrt(sum(∫(e_ex*e_ex)*dΩ_N)) < 1.0e-2

  # A space with Dirichlet DOFs is rejected
  @test_throws ArgumentError FESpace(model,ReferenceFE(lagrangian,Float64,1);
                                     dirichlet_tags="boundary",constraint=:zeromean)

  return true
end

end