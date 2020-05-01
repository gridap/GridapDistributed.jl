module DistributedAssemblersTests

using Gridap
using Gridap.FESpaces
using GridapDistributed
using Test

using GridapDistributed: SparseMatrixAssemblerX
using GridapDistributed: RowsComputedLocally
using SparseArrays

subdomains = (2,3)
comm = SequentialCommunicator(subdomains)

domain = (0,1,0,1)
cells = (10,10)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

V = FESpace(comm,model=model,valuetype=Float64,reffe=:Lagrangian,order=1)

U = TrialFESpace(V)

strategy = RowsComputedLocally(V)

T = Float64
vector_type = Vector{T}
matrix_type = SparseMatrixCSC{T,Int}

assem = SparseMatrixAssemblerX(matrix_type, vector_type, U, V, strategy)

function setup_terms(part,(model,gids))

  trian = Triangulation(model)
  
  degree = 1
  quad = CellQuadrature(trian,degree)

  a(u,v) = v*u
  l(v) = 1*v
  t1 = AffineFETerm(a,l,trian,quad)

  (t1,)
end

terms = DistributedData(setup_terms,model)

#A = assemble_matrix(assem,terms)
b = assemble_vector(assem,terms)

@test sum(b) ≈ 1
#mat = A.mat
#@test ones(1,size(mat,1))*mat*ones(size(mat,2)) ≈ [1]


end # module
