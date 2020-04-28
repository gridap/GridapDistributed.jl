module DistributedFESpacesTests

using Gridap
using GridapDistributed
using Gridap.FESpaces
using Test

comm = SequentialCommunicator()

subdomains = (2,2)
domain = (0,1,0,1)
cells = (4,4)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

nsubdoms = prod(subdomains)

V = FESpace(comm,model=model,valuetype=Float64,reffe=:Lagrangian,order=1)

do_on_parts(get_spaces_and_gids(V),model.models) do part,(space,gids),model

  uh_gids = FEFunction(space,gids.lid_to_gid)
  uh_owner = FEFunction(space,gids.lid_to_owner)
  trian = Triangulation(model)
  writevtk(trian,"results_$(part)",cellfields=["gid"=>uh_gids,"owner"=>uh_owner])
end

#using GridapDistributed: SparseMatrixAssemblerX
#using GridapDistributed: RowsComputedLocally
#using SparseArrays
#
## Define type of vector and matrix
#T = Float64
#vector_type = Vector{T}
#matrix_type = SparseMatrixCSC{T,Int}
#
#b = GloballyAddressableVector{T}(
#  comm, nsubdoms,
#  model.models, V.spaces, V.free_gids) do part, model, V, dof_gids
#
#  U = TrialFESpace(V)
#
#  # FE term to assemble
#  trian = Triangulation(model)
#  quad = CellQuadrature(trian,2)
#  t = FESource( v->1*v, trian, quad)
#
#  # Cell-wise vector
#  dv = get_cell_basis(V)
#  uh0 = zero(U)
#  vecdata = collect_cell_vector(uh0,dv,[t])
#
#  # Strategy for the parallel assembly
#  strategy = RowsComputedLocally(
#    part, dof_gids.lid_to_gid, dof_gids.lid_to_owner)
#
#  # Create the assembler
#  assem = SparseMatrixAssemblerX(
#    matrix_type, vector_type, U, V, strategy, dof_gids, dof_gids)
#
#  # Do the assembly
#  assemble_vector(assem,vecdata...)
#
#end
#
#@test sum(b.vec) ≈ 1

end # module
