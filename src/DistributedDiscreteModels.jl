# @santiagobadia : In this model, gluing is via global VEF dofs, I guess.
# Do we want something else here? Do we need e.g. ghost cells too?
# I agree that with global vef IDs we have a well-defined distributed model,
# so these cells can be created when computing the triangulation and grid,
# because they will certainly be needed for e.g. dG formulations, etc...
struct DistributedDiscreteModel
  models::ScatteredVector{<:DiscreteModel}
  gids::GhostedVector{Int}
end

function Gridap.writevtk(model::DistributedDiscreteModel,filebase::String)

  function task(part,model,gids)
    cdata = ["gids"=>gids.lid_to_gid,"owner"=>gids.lid_to_owner]
    filebase_part = filebase*"_$(part)"
    trian = Triangulation(model)
    writevtk(trian,filebase_part,celldata=cdata)
  end

  do_on_parts(task,model.models,model.gids)

end
