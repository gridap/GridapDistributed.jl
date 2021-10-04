
struct DistributedDiscreteModel{Dc,Dp,A,B}
  models::A
  gids::B
  function DistributedDiscreteModel(
    models::AbstractPData{<:DiscreteModel{Dc,Dp}}, gids::PRange) where {Dc,Dp}
    A = typeof(models)
    B = typeof(gids)
    new{Dc,Dp,A,B}(models,gids)
  end
end

function Geometry.get_face_labeling(model::DistributedDiscreteModel)
  map_parts(model.models) do  model
     get_face_labeling(model)
  end
end

function Geometry.add_tag_from_tags!(labels::AbstractPData{<:FaceLabeling}, name, tags)
  map_parts(labels) do labels
    add_tag_from_tags!(labels, name, tags)
  end
end

function Geometry.CartesianDiscreteModel(
  parts::AbstractPData{<:Integer},args...;kwargs...)

  desc = CartesianDescriptor(args...; kwargs...)
  nc = desc.partition
  @assert length(size(parts)) == length(nc) "A CartesianDiscreteModel needs a Cartesian subdomain partition"
  gcids = PCartesianIndices(parts,nc,PArrays.with_ghost)
  gids = PRange(parts,nc,PArrays.with_ghost)
  models = map_parts(parts,gcids) do part, gcids
    cmin = first(gcids)
    cmax = last(gcids)
    CartesianDiscreteModel(desc,cmin,cmax)
  end
  DistributedDiscreteModel(models,gids)
end
