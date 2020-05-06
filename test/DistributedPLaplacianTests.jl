module DistributedPLaplacianTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed

using GridapDistributed: SparseMatrixAssemblerX
using GridapDistributed: RowsComputedLocally
using SparseArrays
using LinearAlgebra: norm

# Select matrix and vector types for discrete problem
# Note that here we use serial vectors and matrices
# but the assembly is distributed
T = Float64
vector_type = Vector{T}
matrix_type = SparseMatrixCSC{T,Int}

# Manufactured solution
u(x) = x[1] + x[2] + 1
f(x) = - Δ(u)(x)

const p = 3
@law flux(∇u) = norm(∇u)^(p-2) * ∇u
@law dflux(∇du,∇u) = (p-2)*norm(∇u)^(p-4)*inner(∇u,∇du)*∇u + norm(∇u)^(p-2)*∇du

# Discretization
subdomains = (2,2)
domain = (0,1,0,1)
cells = (4,4)
comm = SequentialCommunicator(subdomains)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

# FE Spaces
order = 3
V = FESpace(
  vector_type, valuetype=Float64, reffe=:Lagrangian, order=order,
  model=model, conformity=:H1, dirichlet_tags="boundary")

U = TrialFESpace(V,u)

# Terms in the weak form
terms = DistributedData(model) do part, (model,gids)

  trian = Triangulation(model)
  
  degree = 2*order
  quad = CellQuadrature(trian,degree)

  res(u,v) = inner( ∇(v), flux(∇(u)) ) - inner(v,f)
  jac(u,du,v) = inner(  ∇(v) , dflux(∇(du),∇(u)) )
  t_Ω = FETerm(res,jac,trian,quad)

  (t_Ω,)
end

# Chose parallel assembly strategy
strategy = RowsComputedLocally(V)

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


end # module
