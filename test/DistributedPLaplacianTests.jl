module DistributedPLaplacianTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed
using SparseArrays
using LinearAlgebra: norm

function run(comm,subdomains,assembly_strategy::AbstractString, global_dofs::Bool)
  # Select matrix and vector types for discrete problem
  # Note that here we use serial vectors and matrices
  # but the assembly is distributed
  T = Float64
  vector_type = Vector{T}
  matrix_type = SparseMatrixCSC{T,Int}

  # Manufactured solution
  u(x) = x[1] + x[2] + 1
  f(x) = - Δ(u)(x)

  p = 3
  @law flux(∇u) = norm(∇u)^(p-2) * ∇u
  @law dflux(∇du,∇u) = (p-2)*norm(∇u)^(p-4)*(∇u⋅∇du)*∇u + norm(∇u)^(p-2)*∇du

  # Discretization

  domain = (0,1,0,1)
  cells = (4,4)
  model = CartesianDiscreteModel(comm,subdomains,domain,cells)

  # FE Spaces
  order = 3
  V = FESpace(
  vector_type, valuetype=Float64, reffe=:Lagrangian, order=order,
  model=model, conformity=:H1, dirichlet_tags="boundary")

  U = TrialFESpace(V,u)

  # Choose parallel assembly strategy
  if (assembly_strategy == "RowsComputedLocally")
    strategy = RowsComputedLocally(V;global_dofs=global_dofs)
  elseif (assembly_strategy == "OwnedCellsStrategy"; globals_dofs=global_dofs)
    strategy = OwnedCellsStrategy(model, V; global_dofs=global_dofs)
  else
    @assert false "Unknown AssemblyStrategy: $(assembly_strategy)"
  end

  # Terms in the weak form
  terms = DistributedData(model,strategy) do part, (model,gids), strategy
    trian = Triangulation(strategy,model)
    degree = 2*order
    quad = CellQuadrature(trian,degree)
    res(u,v) = ∇(v)⋅flux(∇(u)) - v*f
    jac(u,du,v) = ∇(v)⋅dflux(∇(du),∇(u))
    t_Ω = FETerm(res,jac,trian,quad)
    (t_Ω,)
  end

  # Assembler
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)

  # Non linear solver
  nls = NLSolver(show_trace=true, method=:newton)
  solver = FESolver(nls)

  # FE solution
  op = FEOperator(assem,terms)
  x = rand(T,num_free_dofs(U))
  uh0 = FEFunction(U,x)
  uh, = solve!(uh0,solver,op)

  # Error norms and print solution
  sums = DistributedData(model,uh) do part, (model,gids), uh
    trian = Triangulation(model)
    owned_trian = remove_ghost_cells(trian,part,gids)
    owned_quad = CellQuadrature(owned_trian,2*order)
    owned_uh = restrict(uh,owned_trian)
    #writevtk(owned_trian,"results_plaplacian_$part",cellfields=["uh"=>owned_uh])
    e = u - owned_uh
    l2(u) = u*u
    sum(integrate(l2(e),owned_trian,owned_quad))
  end
  e_l2 = sum(gather(sums))
  tol = 1.0e-9
  @test e_l2 < tol
end 

subdomains = (2,2)
SequentialCommunicator(subdomains) do comm
  run(comm,subdomains,"RowsComputedLocally",true)
  run(comm,subdomains,"OwnedCellsStrategy",true)
end 

end # module
