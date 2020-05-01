module DistributedAssemblersTests

using Gridap
using Gridap.FESpaces
using GridapDistributed
using Test
using SparseArrays

subdomains = (2,2)
comm = SequentialCommunicator(subdomains)

domain = (0,1,0,1)
cells = (4,4)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

V = FESpace(comm,model=model,valuetype=Float64,reffe=:Lagrangian,order=1)

U = TrialFESpace(V)

strategy = RowsComputedLocally(V)

T = Float64
vector_type = Vector{T}
matrix_type = SparseMatrixCSC{T,Int}

assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, strategy)

function setup_terms(part,(model,gids))

  trian = Triangulation(model)
  
  degree = 2
  quad = CellQuadrature(trian,degree)

  a(u,v) = v*u
  l(v) = 1*v
  t1 = AffineFETerm(a,l,trian,quad)

  (t1,)
end

terms = DistributedData(setup_terms,model)

A = assemble_matrix(assem,terms)
b = assemble_vector(assem,terms)

@test sum(b) ≈ 1
@test ones(1,size(A,1))*A*ones(size(A,2)) ≈ [1]

end # module
