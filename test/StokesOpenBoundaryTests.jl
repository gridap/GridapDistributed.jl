module StokesOpenBoundaryTests

using Gridap
using LinearAlgebra
using Test
using Gridap.ODEs, Gridap.Algebra
using GridapDistributed
using PartitionedArrays

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  θ = 0.5

  ut(t) = x -> VectorValue(x[1],x[2])*t
  u = TimeSpaceFunction(ut)

  pt(t) = x -> (x[1]-x[2])*t
  p = TimeSpaceFunction(pt)
  q(x) = t -> p(t,x)

  ft(t) = x -> ∂t(u)(t,x) - Δ(u)(t,x) + ∇(p)(t,x)
  gt(t) = x -> (∇⋅u)(t,x)
  ht(t) = x -> ∇(u)(t,x)⋅VectorValue(0.0,1.0) - p(t,x)*VectorValue(0.0,1.0)
  f = TimeSpaceFunction(ft)
  g = TimeSpaceFunction(gt)
  h = TimeSpaceFunction(ht)

  domain = (0,1,0,1)
  partition = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,partition)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"neumann",6)
  add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,7,8])

  order = 2

  reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  V0 = FESpace(
    model,
    reffeᵤ,
    conformity=:H1,
    dirichlet_tags="dirichlet"
  )

  reffeₚ = ReferenceFE(lagrangian,Float64,order-1)
  Q = TestFESpace(
    model,
    reffeₚ,
    conformity=:H1
  )

  U = TransientTrialFESpace(V0,u)

  P = TrialFESpace(Q)

  Ω = Triangulation(model)
  degree = 2*order
  dΩ = Measure(Ω,degree)

  Γ = Boundary(model,tags=["neumann"])
  dΓ = Measure(Γ,degree)

  #
  a(t,u,v) = ∫(∇(u)⊙∇(v))dΩ
  b(t,(v,q)) = ∫(v⋅f(t))dΩ + ∫(q*g(t))dΩ + ∫(v⋅h(t))dΓ
  m(t,ut,v) = ∫(ut⋅v)dΩ

  X = TransientMultiFieldFESpace([U,P])
  Y = MultiFieldFESpace([V0,Q])

  res(t,(u,p),(v,q)) = a(t,u,v) + m(t,∂t(u),v) - ∫((∇⋅v)*p)dΩ + ∫(q*(∇⋅u))dΩ - b(t,(v,q))
  jac(t,(u,p),(du,dp),(v,q)) = a(t,du,v) - ∫((∇⋅v)*dp)dΩ + ∫(q*(∇⋅du))dΩ
  jac_t(t,(u,p),(dut,dpt),(v,q)) = m(t,dut,v)

  U0 = U(0.0)
  P0 = P(0.0)
  X0 = X(0.0)
  uh0 = interpolate_everywhere(u(0.0),U0)
  ph0 = interpolate_everywhere(p(0.0),P0)
  xh0 = interpolate_everywhere([uh0,ph0],X0)

  op = TransientFEOperator(res,(jac,jac_t),X,Y)

  t0 = 0.0
  tF = 1.0
  dt = 0.1

  ls  = LUSolver()
  nls = NewtonRaphsonSolver(ls,1.0e-6,10)
  ode_solver = ThetaMethod(nls,dt,θ)

  sol_t = solve(ode_solver,op,t0,tF,xh0)

  l2(w) = w⋅w
  tol = 1.0e-6
  for (tn, xh_tn) in sol_t
    uh_tn = xh_tn[1]
    ph_tn = xh_tn[2]
    #writevtk(Ω,"output/tmp_stokes_OB_sol_$tn.vtu",cellfields=["u"=>uh_tn,"p"=>ph_tn])
    e = u(tn) - uh_tn
    el2 = sqrt(sum(∫(l2(e))dΩ))
    e = p(tn) - ph_tn
    el2 = sqrt(sum(∫(l2(e))dΩ))
    @test el2 < tol
  end
end

end #module
