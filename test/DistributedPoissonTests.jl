module DistributedPoissonTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays

function run(comm,subdomains)
  # Manufactured solution
  u(x) = x[1] + x[2]
  f(x) = -Δ(u)(x)

  # Discretization
  domain = (0, 1, 0, 1)
  cells = (4, 4)
  model = CartesianDiscreteModel(comm, subdomains, domain, cells)

  # FE Spaces
  order=1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(model=model,
              reffe=reffe,
              conformity=:H1,
              dirichlet_tags="boundary")
  U = TrialFESpace(V, u)

  trian=Triangulation(model)
  dΩ=Measure(trian,2*(order+1))

  function a(u,v)
    ∫(∇(v)⋅∇(u))dΩ
  end
  function l(v)
    ∫(v*f)dΩ
  end

  # FE solution
  op = AffineFEOperator(a,l,U,V)
  uh = solve(op)

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
