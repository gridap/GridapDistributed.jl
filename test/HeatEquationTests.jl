module HeatEquationTests

using Gridap
using Gridap.ODEs
using GridapDistributed
using PartitionedArrays
using Test

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  θ = 0.2

  u(x,t) = (1.0-x[1])*x[1]*(1.0-x[2])*x[2]*t
  u(t::Real) = x -> u(x,t)
  f(t) = x -> ∂t(u)(x,t)-Δ(u(t))(x)

  domain = (0,1,0,1)
  partition = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,partition)

  order = 2

  reffe = ReferenceFE(lagrangian,Float64,order)
  V0 = FESpace(
    model,
    reffe,
    conformity=:H1,
    dirichlet_tags="boundary"
  )
  U = TransientTrialFESpace(V0,u)

  Ω = Triangulation(model)
  degree = 2*order
  dΩ = Measure(Ω,degree)

  #
  m(u,v) = ∫(u*v)dΩ
  a(u,v) = ∫(∇(v)⋅∇(u))dΩ
  b(t,v) = ∫(v*f(t))dΩ

  res(t,u,v) = a(u,v) + m(∂t(u),v) - b(t,v)
  jac(t,u,du,v) = a(du,v)
  jac_t(t,u,dut,v) = m(dut,v)

  op = TransientFEOperator(res,jac,jac_t,U,V0)
  op_constant = TransientConstantMatrixFEOperator(m,a,b,U,V0)

  t0 = 0.0
  tF = 1.0
  dt = 0.1

  U0 = U(0.0)
  uh0 = interpolate_everywhere(u(0.0),U0)

  ls = LUSolver()
  ode_solver = ThetaMethod(ls,dt,θ)

  sol_t = solve(ode_solver,op,uh0,t0,tF)
  sol_t_const = solve(ode_solver,op_constant,uh0,t0,tF)

  l2(w) = w*w

  tol = 1.0e-6

  for (uh_tn, tn) in sol_t
    e = u(tn) - uh_tn
    el2 = sqrt(sum( ∫(l2(e))dΩ ))
    @test el2 < tol
  end

  for (uh_tn, tn) in sol_t_const
    e = u(tn) - uh_tn
    el2 = sqrt(sum( ∫(l2(e))dΩ ))
    @test el2 < tol
  end
end

end #module
