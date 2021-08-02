module DistributedPLaplacianTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays
using LinearAlgebra: norm

function run(comm,subdomains)
  # Manufactured solution
  u(x) = x[1] + x[2] + 1
  f(x) = - Δ(u)(x)

  p = 3
  flux(∇u) = norm(∇u)^(p-2) * ∇u
  dflux(∇du,∇u) = (p-2)*norm(∇u)^(p-4)*(∇u⋅∇du)*∇u + norm(∇u)^(p-2)*∇du

  # Discretization

  domain = (0,1,0,1)
  cells = (4,4)
  model = CartesianDiscreteModel(comm,subdomains,domain,cells)

  # FE Spaces
  order=3
  degree=2*order
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(model=model,
              reffe=reffe,
              conformity=:H1,
              dirichlet_tags="boundary")
  U = TrialFESpace(V,u)

  trian=Triangulation(model)
  dΩ=Measure(trian,degree)

  function res(u,v)
     ∫(∇(v)⋅(flux∘∇(u)))*dΩ
  end
  function jac(u,du,v)
     ∫( ∇(v)⋅(dflux∘(∇(du),∇(u))) )*dΩ
  end

  # Non linear solver
  nls = NLSolver(show_trace=true, method=:newton)
  solver = FESolver(nls)

  # FE solution
  op = FEOperator(res,jac,U,V)
  x = rand(num_free_dofs(U))
  uh0 = FEFunction(U,x)
  uh, = solve!(uh0,solver,op)

  # Error norms and print solution
  trian=Triangulation(OwnedCells,model)
  dΩ=Measure(trian,2*order)
  e = u-uh
  e_l2 = sum(∫(e*e)dΩ)
  tol = 1.0e-9
  println("$(e_l2) < $(tol)")
  @test e_l2 < tol
end

subdomains = (2,2)
SequentialCommunicator(subdomains) do comm
  run(comm,subdomains)
end

end # module
