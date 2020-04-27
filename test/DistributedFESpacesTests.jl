module DistributedFESpacesTests

using Gridap
using GridapDistributed
using Gridap.FESpaces

comm = SequentialCommunicator()

subdomains = (2,2)
domain = (0,1,0,1)
cells = (4,4)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

V = FESpace(comm,model=model,valuetype=Float64,reffe=:Lagrangian,order=1)

function print_dofs(part,lspace,free_gids,model)
  uh_gids = FEFunction(lspace,free_gids.lid_to_gid)
  uh_owner = FEFunction(lspace,free_gids.lid_to_owner)
  trian = Triangulation(model)
  writevtk(trian,"results_$(part)",cellfields=["gid"=>uh_gids,"owner"=>uh_owner])
end

do_on_parts(print_dofs,V.spaces,V.free_gids,model.models)

using GridapDistributed: GloballyAddressableVector
using GridapDistributed: num_parts
using GridapDistributed: SequentialGloballyAddressableVectorPart
using GridapDistributed: SparseMatrixAssemblerX
using GridapDistributed: RowsComputedLocally

function init_vectors(part,lspace,gids)
  vec = zeros(Float64,gids.ngids)
  SequentialGloballyAddressableVectorPart(vec)
end

b = GloballyAddressableVector{Float64}(init_vectors,comm,num_parts(V.spaces),V.spaces,V.free_gids)

function assem_vector(part,lmodel,lspace, dof_gids, cell_gids, lb)

  lV = lspace
  lU = TrialFESpace(lspace)

  dv = get_cell_basis(lV)
  uh0 = zero(lU)

  trian = Triangulation(lmodel)
  quad = CellQuadrature(trian,2)

  t = FESource( v->1*v, trian, quad)

  vecdata = collect_cell_vector(uh0,dv,[t])

  lid_to_gid = dof_gids.lid_to_gid
  cell_to_owner = cell_gids.lid_to_owner
  lid_to_owner = dof_gids.lid_to_owner

  strategy = RowsComputedLocally(lid_to_gid,cell_to_owner,lid_to_owner)

  assem = SparseMatrixAssemblerX(lU,lV,strategy)
  assemble_vector!(lb,assem,vecdata...)
end

do_on_parts(assem_vector,model.models,V.spaces,V.free_gids,model.gids,b)


end # module
