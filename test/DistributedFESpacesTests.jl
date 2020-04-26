module DistributedFESpacesTests

using Gridap
using GridapDistributed

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

end # module
