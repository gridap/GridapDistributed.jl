"""
"""
struct DistributedVisualizationData{A<:AbstractArray}
  visdata::A
end

local_views(d::DistributedVisualizationData) = d.visdata

function Base.getproperty(x::DistributedVisualizationData, sym::Symbol)
  if sym == :grid
    map(i->i.grid,x.visdata)
  elseif sym == :filebase
    r=nothing
    map(x.visdata) do visdata
      r = visdata.filebase
    end
    r
  elseif sym == :celldata
    map(i->i.celldata,x.visdata)
  elseif sym == :nodaldata
    map(i->i.nodaldata,x.visdata)
  else
    getfield(x, sym)
  end
end

function Base.propertynames(x::DistributedVisualizationData, private::Bool=false)
  (fieldnames(typeof(x))...,:grid,:filebase,:celldata,:nodaldata)
end

# Define how each object is visualized

function Visualization.visualization_data(
  model::DistributedDiscreteModel{Dc},
  filebase::AbstractString;
  labels=get_face_labeling(model)) where Dc

  cell_gids = get_cell_gids(model)
  vd = map(local_views(model),partition(cell_gids),labels.labels) do model,gids,labels
    part = part_id(gids)
    vd = visualization_data(model,filebase;labels=labels)
    vd_cells = vd[end]
    push!(vd_cells.celldata, "gid" => local_to_global(gids))
    push!(vd_cells.celldata, "part" => local_to_owner(gids))
    vd
  end
  r = []
  for i in 0:Dc
    push!(r,DistributedVisualizationData(map(x->x[i+1],vd)))
  end
  r
end

function Visualization.visualization_data(
  trian::DistributedTriangulation,
  filebase::AbstractString;
  order=-1,
  nsubcells=-1,
  celldata=nothing,
  cellfields=nothing)

  trians    = trian.trians
  cell_gids = get_cell_gids(trian.model)

  cdat = _prepare_cdata(trians,celldata)
  fdat = _prepare_fdata(trians,cellfields)

  vd = map(
    partition(cell_gids),trians,cdat,fdat) do lindices,trian,celldata,cellfields
    part = part_id(lindices)
    _celldata = Dict{Any,Any}(celldata)
    # we do not use "part" since it is likely to be used by the user
    if haskey(_celldata,"piece")
      @unreachable "piece is a reserved cell data name"
    end
    _celldata["piece"] = fill(part,num_cells(trian))
    vd = visualization_data(
      trian,filebase;
      order=order,nsubcells=nsubcells,
      celldata=_celldata,cellfields=cellfields)
    @assert length(vd) == 1
    vd[1]
  end
  [DistributedVisualizationData(vd)]
end

function _prepare_cdata(trians,a::Nothing)
  map(trians) do t
    Dict()
  end
end

function _prepare_cdata(trians,a)
  if length(a) == 0
    return map(trians) do t
      Dict()
    end
  end
  ks = []
  vs = []
  for (k,v) in a
    push!(ks,k)
    push!(vs,v)
  end
  map(vs...) do vs...
    b = []
    for i in 1:length(vs)
      push!(b,ks[i]=>vs[i])
    end
    b
  end
end

function _prepare_fdata(trians,a::Nothing)
  map(trians) do t
    Dict()
  end
end

function _prepare_fdata(trians,a)
  _fdata(v::DistributedCellField,trians) = v.fields
  _fdata(v::AbstractArray,trians) = v
  _fdata(v,trians) = map(ti->v,trians)
  if length(a) == 0
    return map(trians) do t
      Dict()
    end
  end
  ks = []
  vs = []
  for (k,v) in a
    push!(ks,k)
    push!(vs,_fdata(v,trians))
  end
  map(vs...) do vs...
    b = []
    for i in 1:length(vs)
      push!(b,ks[i]=>vs[i])
    end
    b
  end
end

# Vtk related

function Visualization.write_vtk_file(
  parts::AbstractArray,
  grid::AbstractArray{<:Grid}, filebase; celldata, nodaldata)
  pvtk = Visualization.create_vtk_file(parts,grid,filebase;celldata=celldata,nodaldata=nodaldata)
  map(vtk_save,pvtk)
end

function Visualization.create_vtk_file(
  parts::AbstractArray,
  grid::AbstractArray{<:Grid}, 
  filebase; 
  celldata, nodaldata)
  nparts = length(parts)
  map(parts,grid,celldata,nodaldata) do part,g,c,n
    Visualization.create_pvtk_file(
      g,filebase;
      part=part,nparts=nparts,
      celldata=c,nodaldata=n)
  end
end

const DistributedModelOrTriangulation = Union{DistributedDiscreteModel,DistributedTriangulation}

function Visualization.writevtk(arg::DistributedModelOrTriangulation,args...;kwargs...)
  parts=get_parts(arg)
  map(visualization_data(arg,args...;kwargs...)) do visdata
    write_vtk_file(
    parts,visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata)
  end
end

function Visualization.createvtk(arg::DistributedModelOrTriangulation,args...;kwargs...)
  v = visualization_data(arg,args...;kwargs...)
  parts=get_parts(arg)
  @notimplementedif length(v) != 1
  visdata = first(v)
  Visualization.create_vtk_file(
    parts,visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata)
end

struct DistributedPvd{T<:AbstractArray}
  pvds::T
end

function Visualization.createpvd(parts::AbstractArray,args...;kwargs...)
  pvds = map_main(parts) do part
    paraview_collection(args...;kwargs...)
  end
  DistributedPvd(pvds)
end

function Visualization.createpvd(f,parts::AbstractArray,args...;kwargs...)
  pvd = createpvd(parts,args...;kwargs...)
  try
    f(pvd)
  finally
    savepvd(pvd)
  end
end

function Visualization.savepvd(pvd::DistributedPvd)
  map_main(pvd.pvds) do pvd
    vtk_save(pvd)
  end
end

function Base.setindex!(pvd::DistributedPvd,pvtk::AbstractArray,time::Real)
  map(vtk_save,pvtk)
  map_main(pvtk,pvd.pvds) do pvtk,pvd
    pvd[time] = pvtk
  end
end
