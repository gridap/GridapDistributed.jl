struct DistributedVisualizationData{A<:AbstractPData}
  visdata::A
end

function Base.getproperty(x::DistributedVisualizationData, sym::Symbol)
  if sym == :grid
    map_parts(i->i.grid,x.visdata)
  elseif sym == :filebase
    map_parts(i->i.filebase,x.visdata)
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

  parts = get_part_ids(model.models)
  nparts = length(parts)
  vd = map_parts(
    parts,model.models,model.gids.partition,labels) do part,model,gids,labels

    n = lpad(part,ceil(Int,log10(nparts)),'0')
    vd = visualization_data(model,"$(filebase)_$(n)";labels=labels)
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
  celldata=_empty_data(trian.trians),
  cellfields=_empty_data(trian.trians))

  trians = trian.trians
  parts = get_part_ids(trians)
  nparts = length(trians)

  vd = map_parts(
    parts,trians,celldata,cellfields) do part,trian,celldata,cellfields
    n = lpad(part,ceil(Int,log10(nparts)),'0')
    vd = visualization_data(
      trian,"$(filebase)_$(n)";
      order=order,nsubcells=nsubcells,
      celldata=celldata,cellfields=cellfields)
    @assert length(vd) == 1
    vd[1]
  end
  [DistributedVisualizationData(vd)]
end

function _empty_data(a)
  parts = get_part_ids(a)
  map_parts(i->Dict(),parts)
end

# Vtk related

# TODO use pvd format

function Visualization.write_vtk_file(
  grid::AbstractPData{<:Grid}, filebase; celldata, nodaldata)
  map_parts(grid,filebase,celldata,nodaldata) do g,f,c,n
    write_vtk_file(g,f;celldata=c,nodaldata=n)
  end
end

function Visualization.create_vtk_file(
  grid::AbstractPData{<:Grid}, filebase; celldata, nodaldata)
  map_parts(grid,filebase,celldata,nodaldata) do g,f,c,n
    create_vtk_file(g,f;celldata=c,nodaldata=n)
  end
end

