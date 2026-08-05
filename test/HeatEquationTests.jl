module HeatEquationTests

using Gridap
using Gridap.ODEs, Gridap.Algebra
using GridapDistributed
using PartitionedArrays
using Test

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  θ = 0.2

  ut(t) = x -> (1.0-x[1])*x[1]*(1.0-x[2])*x[2]*t
  u = TimeSpaceFunction(ut)
  ft(t) = x -> ∂t(u)(t,x) - Δ(u)(t,x)
  f = TimeSpaceFunction(ft)

  domain = (0,1,0,1)
  model = CartesianDiscreteModel(ranks,parts,domain,(4,4))

  order = 2
  reffe = ReferenceFE(lagrangian,Float64,order)
  V0 = FESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
  U = TransientTrialFESpace(V0,u)

  Ω = Triangulation(model)
  degree = 2*order
  dΩ = Measure(Ω,degree)

  m(t,u,v) = ∫(u*v)dΩ
  a(t,u,v) = ∫(∇(v)⋅∇(u))dΩ
  b(t,v) = ∫(v*f(t))dΩ

  res(t,u,v) = a(t,u,v) + m(t,∂t(u),v) - b(t,v)
  jac(t,u,du,v) = a(t,du,v)
  jac_t(t,u,dut,v) = m(t,dut,v)

  op = TransientFEOperator(res,jac,jac_t,U,V0)

  assembler = SparseMatrixAssembler(U,V0,Assembled())
  op_constant = TransientLinearFEOperator(
    (a,m),b,U,V0,constant_forms=(true,true),assembler=assembler
  )

  t0 = 0.0
  tF = 1.0
  dt = 0.1

  U0 = U(0.0)
  uh0 = interpolate_everywhere(u(0.0),U0);

  ls = LUSolver()
  linear_ode_solver = ThetaMethod(ls,dt,θ)
  sol_t_const = solve(linear_ode_solver,op_constant,t0,tF,uh0)

  nls = NewtonRaphsonSolver(ls,1.0e-6,10)
  nonlinear_ode_solver = ThetaMethod(nls,dt,θ)
  sol_t = solve(nonlinear_ode_solver,op,t0,tF,uh0)

  l2(w) = w*w
  tol = 1.0e-6
  for (tn, uh_tn) in sol_t
    e = u(tn) - uh_tn
    el2 = sqrt(sum( ∫(l2(e))dΩ ))
    @test el2 < tol
  end

  for (tn, uh_tn) in sol_t_const
    e = u(tn) - uh_tn
    el2 = sqrt(sum( ∫(l2(e))dΩ ))
    @test el2 < tol
  end
end

end #module
