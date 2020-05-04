module DistributedPoissonTests

using Test
using Gridap
using Gridap.FESpaces
using GridapDistributed

using GridapDistributed: SparseMatrixAssemblerX
using GridapDistributed: RowsComputedLocally
using SparseArrays

# Manufactured solution
u(x) = x[1] + x[2]
#f(x) = - Δ(u)(x)
f(x) = u(x)

# Discretization
subdomains = (2,2)
domain = (0,1,0,1)
cells = (4,4)
comm = SequentialCommunicator(subdomains)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

# FE Spaces
order = 1
V = FESpace(
  comm, valuetype=Float64, reffe=:Lagrangian, order=1,
  model=model, conformity=:H1)# TODO, dirichlet_tags="boundary") for the moment we solve a l2 problem

U = TrialFESpace(V,u)

# Terms in the weak form
terms = DistributedData(model) do part, (model,gids)

  trian = Triangulation(model)
  
  degree = 2*order
  quad = CellQuadrature(trian,degree)

  #a(u,v) = ∇(v)*∇(u)
  a(u,v) = v*u
  l(v) = v*f
  t1 = AffineFETerm(a,l,trian,quad)

  (t1,)
end

# Select matrix and vector types for discrete problem
# Note that here we use serial vectors and matrices
# but the assembly is distributed
T = Float64
vector_type = Vector{T}
matrix_type = SparseMatrixCSC{T,Int}

# Chose parallel assembly strategy
strategy = RowsComputedLocally(V)

# Assembly
assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)
op = AffineFEOperator(assem,terms)
A = get_matrix(op)
b = get_vector(op)

# FE solution
x = A \ b
uh = FEFunction(U,x)

# Error norms and print solution
sums = DistributedData(model,uh) do part, (model,gids), uh

  trian = Triangulation(model)

  owned_trian = remove_ghost_cells(trian,part,gids)
  owned_quad = CellQuadrature(owned_trian,2*order)
  owned_uh = restrict(uh,owned_trian)

  writevtk(owned_trian,"results_$part",cellfields=["uh"=>owned_uh])

  e = u - owned_uh

  l2(u) = u*u

  sum(integrate(l2(e),owned_trian,owned_quad))

end

e_l2 = sum(gather(sums))

tol = 1.0e-9
@test e_l2 < tol


end # module
