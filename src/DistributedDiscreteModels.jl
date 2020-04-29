struct DistributedDiscreteModel
  models::ScatteredVector{<:DiscreteModel}
  gids::GhostedVector{Int}
end

function get_distributed_data(dmodel::DistributedDiscreteModel)
  models = dmodel.models
  gids = dmodel.gids
  comm = get_comm(models)
  nparts = num_parts(models)

  T = Tuple{get_part_type(models),get_part_type(gids)}
  ScatteredVector{T}(comm,nparts,models,gids) do part, model, lgids
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

function Gridap.Triangulation(dmodel::DistributedDiscreteModel,args...)
  comm = get_comm(dmodel)
  nparts = num_parts(dmodel)
  trians = ScatteredVector{Triangulation}(comm,nparts,dmodel.models) do part, model
    Triangulation(model,args...)
  end
  DistributedTriangulation(trians)
end

function Gridap.BoundaryTriangulation(dmodel::DistributedDiscreteModel,args...)
  comm = get_comm(dmodel)
  nparts = num_parts(dmodel)
  trians = ScatteredVector{Triangulation}(comm,nparts,dmodel.models) do part, model
    BoundaryTriangulation(model,args...)
  end
  DistributedTriangulation(trians)
end

function Gridap.SkeletonTriangulation(dmodel::DistributedDiscreteModel,args...)
  comm = get_comm(dmodel)
  nparts = num_parts(dmodel)
  trians = ScatteredVector{Triangulation}(comm,nparts,dmodel.models) do part, model
    SkeletonTriangulation(model,args...)
  end
  DistributedTriangulation(trians)
end

