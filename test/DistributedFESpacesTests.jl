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

end # module
