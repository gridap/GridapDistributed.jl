"""
"""
struct DistributedVisualizationData{A<:AbstractPData}
  visdata::A
end

local_views(d::DistributedVisualizationData) = d.visdata

function Base.getproperty(x::DistributedVisualizationData, sym::Symbol)
  if sym == :grid
    map_parts(i->i.grid,x.visdata)
  elseif sym == :filebase
    get_part(x.visdata).filebase
  elseif sym == :celldata
    map_parts(i->i.celldata,x.visdata)
  elseif sym == :nodaldata
    map_parts(i->i.nodaldata,x.visdata)
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

  parts  = get_parts(model)
  nparts = length(parts)
  cell_gids = get_cell_gids(model)
  vd = map_parts(
    parts,local_views(model),cell_gids.partition,labels.labels) do part,model,gids,labels

    vd = visualization_data(model,filebase;labels=labels)
    vd_cells = vd[end]
    push!(vd_cells.celldata, "gid" => gids.lid_to_gid)
    push!(vd_cells.celldata, "part" => gids.lid_to_part)
    vd
  end
  r = []
  for i in 0:Dc
    push!(r,DistributedVisualizationData(map_parts(x->x[i+1],vd)))
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

  trians = trian.trians
  parts = get_part_ids(trians)
  nparts = length(trians)

  cdat = _prepare_cdata(trians,celldata)
  fdat = _prepare_fdata(trians,cellfields)

  vd = map_parts(
    parts,trians,cdat,fdat) do part,trian,celldata,cellfields
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
  map_parts(trians) do t
    Dict()
  end
end

function _prepare_cdata(trians,a)
  if length(a) == 0
    return map_parts(trians) do t
      Dict()
    end
  end
  ks = []
  vs = []
  for (k,v) in a
    push!(ks,k)
    push!(vs,v)
  end
  map_parts(vs...) do vs...
    b = []
    for i in 1:length(vs)
      push!(b,ks[i]=>vs[i])
    end
    b
  end
end

function _prepare_fdata(trians,a::Nothing)
  map_parts(trians) do t
    Dict()
  end
end

function _prepare_fdata(trians,a)
  _fdata(v::DistributedCellField,trians) = v.fields
  _fdata(v::AbstractPData,trians) = v
  _fdata(v,trians) = map_parts(ti->v,trians)
  if length(a) == 0
    return map_parts(trians) do t
      Dict()
    end
  end
  ks = []
  vs = []
  for (k,v) in a
    push!(ks,k)
    push!(vs,_fdata(v,trians))
  end
  map_parts(vs...) do vs...
    b = []
    for i in 1:length(vs)
      push!(b,ks[i]=>vs[i])
    end
    b
  end
end

# Vtk related

function Visualization.write_vtk_file(
  grid::AbstractPData{<:Grid}, filebase; celldata, nodaldata)
  pvtk = Visualization.create_vtk_file(grid,filebase;celldata=celldata,nodaldata=nodaldata)
  map_parts(vtk_save,pvtk)
end

function Visualization.create_vtk_file(
  grid::AbstractPData{<:Grid}, filebase; celldata, nodaldata)
  parts = get_part_ids(grid)
  nparts = length(parts)
  map_parts(parts,grid,celldata,nodaldata) do part,g,c,n
    Visualization.create_pvtk_file(
      g,filebase;
      part=part,nparts=nparts,
      celldata=c,nodaldata=n)
  end
end

struct DistributedPvd{T<:AbstractPData}
  pvds::T
end

function Visualization.createpvd(parts::AbstractPData,args...;kwargs...)
  pvds = map_main(parts) do part
    paraview_collection(args...;kwargs...)
  end
  DistributedPvd(pvds)
end

function Visualization.createpvd(f,parts::AbstractPData,args...;kwargs...)
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

function Base.setindex!(pvd::DistributedPvd,pvtk::AbstractPData,time::Real)
  map_parts(vtk_save,pvtk)
  map_main(pvtk,pvd.pvds) do pvtk,pvd
    pvd[time] = pvtk
  end
end
