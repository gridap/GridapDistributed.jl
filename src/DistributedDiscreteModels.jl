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

function Gridap.Geometry.get_face_labeling(model::DistributedDiscreteModel)
  DistributedData(model) do part, (model,gids)
     get_face_labeling(model)
  end
end

"""
    add_tag_from_tags!(lab::FaceLabeling, name::String, tags::Vector{Int})
    add_tag_from_tags!(lab::FaceLabeling, name::String, tags::Vector{String})
    add_tag_from_tags!(lab::FaceLabeling, name::String, tag::Int)
    add_tag_from_tags!(lab::FaceLabeling, name::String, tag::String)
"""
function Gridap.Geometry.add_tag_from_tags!(lab::DistributedData{<:FaceLabeling}, name::String, tags::Vector{Int})
  do_on_parts(lab) do part, lab
    add_tag_from_tags!(lab, name, tags)
  end
end

function Gridap.Geometry.add_tag_from_tags!(
  labels::DistributedData{<:FaceLabeling}, name::String, names::Vector{String})
  @notimplemented
end

function Gridap.Geometry.add_tag_from_tags!(labels::DistributedData{<:FaceLabeling}, name::String, tag::Int)
  @notimplemented
end

function Gridap.Geometry.add_tag_from_tags!(labels::DistributedData{<:FaceLabeling}, name::String, tag::String)
  @notimplemented
end
