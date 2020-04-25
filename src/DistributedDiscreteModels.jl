
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

