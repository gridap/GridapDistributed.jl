module DistributedFESpacesTests

using Gridap
using GridapDistributed
using Gridap.FESpaces
using Test

subdomains = (2,2)
SequentialCommunicator(subdomains) do comm
  domain = (0,1,0,1)
  cells = (4,4)
  model = CartesianDiscreteModel(comm,subdomains,domain,cells)
  nsubdoms = prod(subdomains)
  vector_type = Vector{Float64}
  reffe = ReferenceFE(lagrangian,Float64,1)
  V = FESpace(vector_type,model=model,reffe=reffe)
  do_on_parts(V, model) do part,(space,gids), (model,_)
    uh_gids = FEFunction(space,convert(Vector{Float64},gids.lid_to_gid))
    uh_owner = FEFunction(space,convert(Vector{Float64},gids.lid_to_owner))
    trian = Triangulation(model)
    writevtk(trian,"results_$(part)",cellfields=["gid"=>uh_gids,"owner"=>uh_owner])
  end
end

end # module
