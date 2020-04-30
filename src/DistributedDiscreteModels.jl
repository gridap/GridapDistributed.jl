struct DistributedDiscreteModel
  models::DistributedData{<:DiscreteModel}
  gids::DistributedIndexSet
end

function get_distributed_data(dmodel::DistributedDiscreteModel)
  models = dmodel.models
  gids = dmodel.gids

  DistributedData(models,gids) do part, model, lgids
    model, lgids
  end
end

function Gridap.writevtk(model::DistributedDiscreteModel,filebase::String)

  do_on_parts(model) do part, (model, gids)

    cdata = ["gids"=>gids.lid_to_gid,"owner"=>gids.lid_to_owner]
    filebase_part = filebase*"_$(part)"
    trian = Triangulation(model)
    writevtk(trian,filebase_part,celldata=cdata)
  end

end


