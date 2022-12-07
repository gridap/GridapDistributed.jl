
# We do not inherit from Grid on purpose.
# This object cannot implement the Grid interface in a strict sense
"""
"""
struct DistributedGrid{Dc,Dp,A} <: GridapType
  grids::A
  function DistributedGrid(grids::AbstractPData{<:Grid{Dc,Dp}}) where {Dc,Dp}
    A = typeof(grids)
    new{Dc,Dp,A}(grids)
  end
end

local_views(g::DistributedGrid) = g.grids

function Geometry.OrientationStyle(
  ::Type{<:DistributedGrid{Dc,Dp,A}}) where {Dc,Dp,A}
  OrientationStyle(eltype(A))
end

function Geometry.RegularityStyle(
  ::Type{<:DistributedGrid{Dc,Dp,A}}) where {Dc,Dp,A}
  RegularityStyle(eltype(A))
end

Geometry.num_cell_dims(::DistributedGrid{Dc,Dp}) where {Dc,Dp} = Dc
Geometry.num_cell_dims(::Type{<:DistributedGrid{Dc,Dp}}) where {Dc,Dp} = Dc
Geometry.num_point_dims(::DistributedGrid{Dc,Dp}) where {Dc,Dp} = Dp
Geometry.num_point_dims(::Type{<:DistributedGrid{Dc,Dp}}) where {Dc,Dp} = Dp

# We do not inherit from GridTopology on purpose.
# This object cannot implement the GridTopology interface in a strict sense
"""
"""
struct DistributedGridTopology{Dc,Dp,A} <: GridapType
  topos::A
  function DistributedGridTopology(topos::AbstractPData{<:GridTopology{Dc,Dp}}) where {Dc,Dp}
    A = typeof(topos)
    new{Dc,Dp,A}(topos)
  end
end

local_views(a::DistributedGridTopology) = a.topo

function Geometry.OrientationStyle(
  ::Type{<:DistributedGridTopology{Dc,Dp,A}}) where {Dc,Dp,A}
  OrientationStyle(eltype(A))
end

function Geometry.RegularityStyle(
  ::Type{<:DistributedGridTopology{Dc,Dp,A}}) where {Dc,Dp,A}
  RegularityStyle(eltype(A))
end

Geometry.num_cell_dims(::DistributedGridTopology{Dc,Dp}) where {Dc,Dp} = Dc
Geometry.num_cell_dims(::Type{<:DistributedGridTopology{Dc,Dp}}) where {Dc,Dp} = Dc
Geometry.num_point_dims(::DistributedGridTopology{Dc,Dp}) where {Dc,Dp} = Dp
Geometry.num_point_dims(::Type{<:DistributedGridTopology{Dc,Dp}}) where {Dc,Dp} = Dp

"""
"""
struct DistributedFaceLabeling{A<:AbstractPData{<:FaceLabeling}}
  labels::A
end

local_views(a::DistributedFaceLabeling) = a.labels

function Geometry.add_tag_from_tags!(labels::DistributedFaceLabeling, name, tags)
  map_parts(labels.labels) do labels
    add_tag_from_tags!(labels, name, tags)
  end
end

# We do not inherit from DiscreteModel on purpose.
# This object cannot implement the DiscreteModel interface in a strict sense
"""
"""
struct DistributedDiscreteModel{Dc,Dp,A,B} <: GridapType
  models::A
  face_gids::B
  function DistributedDiscreteModel(
    models::AbstractPData{<:DiscreteModel{Dc,Dp}}, gids::PRange) where {Dc,Dp}
    A = typeof(models)
    face_gids=Vector{PRange}(undef,Dc+1)
    face_gids[Dc+1]=gids
    B = typeof(face_gids)
    new{Dc,Dp,A,B}(models,face_gids)
  end
end

local_views(a::DistributedDiscreteModel) = a.models

Geometry.num_cell_dims(::DistributedDiscreteModel{Dc,Dp}) where {Dc,Dp} = Dc
Geometry.num_cell_dims(::Type{<:DistributedDiscreteModel{Dc,Dp}}) where {Dc,Dp} = Dc
Geometry.num_point_dims(::DistributedDiscreteModel{Dc,Dp}) where {Dc,Dp} = Dp
Geometry.num_point_dims(::Type{<:DistributedDiscreteModel{Dc,Dp}}) where {Dc,Dp} = Dp

function Geometry.num_cells(model::DistributedDiscreteModel{Dc}) where Dc
 num_gids(model.face_gids[Dc+1])
end

function Geometry.num_facets(model::DistributedDiscreteModel{Dc}) where Dc
  _setup_face_gids!(model,Dc-1)
  num_gids(model.face_gids[Dc])
end

function Geometry.num_edges(model::DistributedDiscreteModel{Dc}) where Dc
  _setup_face_gids!(model,1)
  num_gids(model.face_gids[2])
end

function Geometry.num_vertices(model::DistributedDiscreteModel{Dc}) where Dc
  _setup_face_gids!(model,0)
  num_gids(model.face_gids[1])
end

function Geometry.num_faces(model::DistributedDiscreteModel{Dc},dim::Integer) where Dc
  _setup_face_gids!(model,dim)
  num_gids(model.face_gids[dim+1])
end

function Geometry.num_faces(model::DistributedDiscreteModel{Dc}) where Dc
  num_faces=0
  for dim=0:Dc
    _setup_face_gids!(model,dim)
    num_faces+=num_gids(model.face_gids[dim+1])
  end
  num_faces
end

function Geometry.get_grid(model::DistributedDiscreteModel)
  DistributedGrid(map_parts(get_grid,model.models))
end

function Geometry.get_grid_topology(model::DistributedDiscreteModel)
  DistributedGridTopology(map_parts(get_grid_topology,model.models))
end

function Geometry.get_face_labeling(model::DistributedDiscreteModel)
  DistributedFaceLabeling(map_parts(get_face_labeling,model.models))
end

function get_cell_gids(model::DistributedDiscreteModel{Dc}) where Dc
  model.face_gids[Dc+1]
end

function get_face_gids(model::DistributedDiscreteModel,dim::Integer)
  _setup_face_gids!(model,dim)
  model.face_gids[dim+1]
end

# CartesianDiscreteModel

function Geometry.CartesianDiscreteModel(
  parts::AbstractPData{<:Integer},args...;isperiodic=map(i->false,size(parts)),kwargs...)

  desc = CartesianDescriptor(args...;isperiodic=isperiodic,kwargs...)
  nc = desc.partition
  msg = """
  A CartesianDiscreteModel needs a Cartesian subdomain partition
  of the right dimensions.
  """
  @assert length(size(parts)) == length(nc) msg

  if any(isperiodic)
    model = _cartesian_model_with_periodic_bcs(parts,desc)
  else
    gcids = PCartesianIndices(parts,nc,PArrays.with_ghost)
    models = map_parts(parts,gcids) do part, gcids
      cmin = first(gcids)
      cmax = last(gcids)
      CartesianDiscreteModel(desc,cmin,cmax)
    end
    gids = PRange(parts,nc,PArrays.with_ghost)
    model = DistributedDiscreteModel(models,gids)
  end
  model
end

function _cartesian_model_with_periodic_bcs(parts,desc)
  h = desc.sizes
  _origin = map(desc.isperiodic,Tuple(desc.origin),h,size(parts)) do isp,o,h,np
    isp&&np!=1 ? o-h : o
  end
  _sizes = h
  _partition = map(desc.isperiodic,desc.partition,size(parts)) do isp,o,np
    isp&&np!=1 ? o+2 : o
  end
   # Important, the map should be periodic if you want to integrate
   # functions in the ghost cells.
  _map = desc.map
  _isperiodic = map(desc.isperiodic,size(parts)) do isp,np
    np==1 ? isp : false
  end
  _desc = CartesianDescriptor(Point(_origin),_sizes,_partition;map=_map,isperiodic=_isperiodic)
  isperiodic_global = map(desc.isperiodic,size(parts)) do isp,np
    np==1 ? false : isp
  end
  in_bounds = Val(false)
  gcids = PCartesianIndices(parts,desc.partition,PArrays.with_ghost,isperiodic_global,in_bounds)
  nparts = size(parts)
  models = map_parts(parts,gcids) do part, gcids
    cmin = CartesianIndex(map((p,i,n)->( p&&n!=1 ? i+1 : i),desc.isperiodic,Tuple(first(gcids)),nparts))
    cmax = CartesianIndex(map((p,i,n)->( p&&n!=1 ? i+1 : i),desc.isperiodic,Tuple(last(gcids)),nparts))
    remove_boundary = map((p,n)->(p&&n!=1 ? true : false),desc.isperiodic,nparts)
    CartesianDiscreteModel(_desc,cmin,cmax,remove_boundary)
  end
  gids = PRange(parts,desc.partition,PArrays.with_ghost,isperiodic_global)
  model = DistributedDiscreteModel(models,gids)
end

## Helpers to partition a serial model
# Not very scalable but useful in moderate
# cell and proc counts

function compute_cell_graph(model::DiscreteModel,d::Integer=0)
  D = num_cell_dims(model)
  topo = get_grid_topology(model)
  cell_to_dfaces = get_faces(topo,D,d)
  dface_to_cells = get_faces(topo,d,D)
  _cell_graph(cell_to_dfaces,dface_to_cells)
end

function _cell_graph(cell_to_dfaces,dface_to_cells)
  # This can be improved using CSRR format
  ncells = length(cell_to_dfaces)
  c1 = array_cache(cell_to_dfaces)
  c2 = array_cache(dface_to_cells)
  ndata = 0
  for icell in 1:ncells
    ndata += 1
    dfaces = getindex!(c1,cell_to_dfaces,icell)
    for dface in dfaces
      jcells = getindex!(c2,dface_to_cells,dface)
      for jcell in jcells
        if jcell != icell
          ndata += 1
        end
      end
    end
  end
  I = zeros(Int32,ndata)
  J = zeros(Int32,ndata)
  p = 0
  for icell in 1:ncells
    p += 1
    I[p] = icell
    J[p] = icell
    dfaces = getindex!(c1,cell_to_dfaces,icell)
    for dface in dfaces
      jcells = getindex!(c2,dface_to_cells,dface)
      for jcell in jcells
        if jcell != icell
          p += 1
          I[p] = icell
          J[p] = jcell
        end
      end
    end
  end
  m = ncells
  n = ncells
  V = ones(Int8,ndata)
  g = sparse(I,J,V,m,n)
  fill!(g.nzval,Int8(1))
  g
end

function Geometry.DiscreteModel(
  parts::AbstractPData,
  model::DiscreteModel,
  cell_to_part::AbstractArray,
  cell_graph::SparseMatrixCSC = compute_cell_graph(model))

  ncells = num_cells(model)
  @assert length(cell_to_part) == ncells
  @assert size(cell_graph,1) == ncells
  @assert size(cell_graph,2) == ncells

  lcell_to_cell, lcell_to_part, gid_to_part = map_parts(parts) do part
    cell_to_mask = fill(false,ncells)
    icell_to_jcells_ptrs = cell_graph.colptr
    icell_to_jcells_data = cell_graph.rowval
    for icell in 1:ncells
      if cell_to_part[icell] == part
        cell_to_mask[icell] = true
        pini = icell_to_jcells_ptrs[icell]
        pend = icell_to_jcells_ptrs[icell+1]-1
        for p in pini:pend
          jcell = icell_to_jcells_data[p]
          cell_to_mask[jcell] = true
        end
      end
    end
    lcell_to_cell = findall(cell_to_mask)
    lcell_to_part = zeros(Int32,length(lcell_to_cell))
    lcell_to_part .= cell_to_part[lcell_to_cell]
    lcell_to_cell, lcell_to_part, cell_to_part
  end

  partition = map_parts(IndexSet,parts,lcell_to_cell,lcell_to_part)
  exchanger = Exchanger(partition;reuse_parts_rcv=true)
  gids = PRange(ncells,partition,exchanger,gid_to_part)

  models = map_parts(lcell_to_cell) do lcell_to_cell
    DiscreteModelPortion(model,lcell_to_cell)
  end

  DistributedDiscreteModel(models,gids)
end

# Triangulation

# We do not inherit from Triangulation on purpose.
# This object cannot implement the Triangulation interface in a strict sense
"""
"""
struct DistributedTriangulation{Dc,Dp,A,B} <: GridapType
  trians::A
  model::B
  function DistributedTriangulation(
    trians::AbstractPData{<:Triangulation{Dc,Dp}},
    model::DistributedDiscreteModel) where {Dc,Dp}
    A = typeof(trians)
    B = typeof(model)
    new{Dc,Dp,A,B}(trians,model)
  end
end

local_views(a::DistributedTriangulation) = a.trians

Geometry.num_cell_dims(::DistributedTriangulation{Dc,Dp}) where {Dc,Dp} = Dc
Geometry.num_cell_dims(::Type{<:DistributedTriangulation{Dc,Dp}}) where {Dc,Dp} = Dc
Geometry.num_point_dims(::DistributedTriangulation{Dc,Dp}) where {Dc,Dp} = Dp
Geometry.num_point_dims(::Type{<:DistributedTriangulation{Dc,Dp}}) where {Dc,Dp} = Dp

function Geometry.get_background_model(a::DistributedTriangulation)
  a.model
end

# Triangulation constructors

function Geometry.Triangulation(
  model::DistributedDiscreteModel;kwargs...)
  D=num_cell_dims(model)
  Triangulation(no_ghost,ReferenceFE{D},model;kwargs...)
end

function Geometry.BoundaryTriangulation(
  model::DistributedDiscreteModel;kwargs...)
  BoundaryTriangulation(no_ghost,model;kwargs...)
end

function Geometry.SkeletonTriangulation(
  model::DistributedDiscreteModel;kwargs...)
  SkeletonTriangulation(no_ghost,model;kwargs...)
end

function Geometry.Triangulation(
  portion,::Type{ReferenceFE{Dt}},model::DistributedDiscreteModel{Dm};kwargs...) where {Dt,Dm}
  # Generate global ordering for the faces of dimension Dt (if needed)
  gids=get_face_gids(model,Dt)
  trians = map_parts(model.models,gids.partition) do model, gids
    Triangulation(portion,gids,ReferenceFE{Dt},model;kwargs...)
  end
  dtrian=DistributedTriangulation(trians,model)
end

# function Geometry.BoundaryTriangulation(trian::DistributedTriangulation)
#   BoundaryTriangulation(no_ghost,trian)
# end

# function Geometry.BoundaryTriangulation(
#   portion,trian::DistributedTriangulation)
#   trians = map_parts(BoundaryTriangulation,trian.trians)
#   Dc = num_cell_dims(trian.model)
#   gids = trian.model.face_gids[Dc+1]
#   trians2 = map_parts(trians,gids.partition) do trian,gids
#      filter_cells_when_needed(portion,gids,trian)
#   end
#   DistributedTriangulation(trians2,trian.model)
# end

function Geometry.BoundaryTriangulation(
  portion,model::DistributedDiscreteModel{Dc};kwargs...) where Dc
  trians = map_parts(model.models,model.face_gids[Dc+1].partition) do model,gids
    BoundaryTriangulation(portion,gids,model;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

function Geometry.SkeletonTriangulation(trian::DistributedTriangulation)
  SkeletonTriangulation(no_ghost,trian)
end

function Geometry.SkeletonTriangulation(
  portion,trian::DistributedTriangulation)
  trians = map_parts(SkeletonTriangulation,trian.trians)
  Dc = num_cell_dims(trian.model)
  gids = trian.model.face_gids[Dc+1]
  trians2 = map_parts(trians,gids.partition) do trian,gids
     filter_cells_when_needed(portion,gids,trian)
  end
  DistributedTriangulation(trians2,trian.model)
end

function Geometry.SkeletonTriangulation(
  portion,model::DistributedDiscreteModel{Dc};kwargs...) where Dc
  trians = map_parts(model.models,model.face_gids[Dc+1].partition) do model,gids
    SkeletonTriangulation(portion,gids,model;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

function Geometry.Triangulation(
  portion,gids::AbstractIndexSet, args...;kwargs...) where D
  trian = Triangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.BoundaryTriangulation(
  portion,gids::AbstractIndexSet,args...;kwargs...)
  trian = BoundaryTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.SkeletonTriangulation(
  portion,gids::AbstractIndexSet,args...;kwargs...)
  trian = SkeletonTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.InterfaceTriangulation(
  portion,gids::AbstractIndexSet,args...;kwargs...)
  trian = InterfaceTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.InterfaceTriangulation(a::DistributedTriangulation,b::DistributedTriangulation)
  trians = map_parts(InterfaceTriangulation,a.trians,b.trians)
  @assert a.model === b.model
  DistributedTriangulation(trians,a.model)
end

function Geometry.Triangulation(
  portion, model::DistributedDiscreteModel;kwargs...)
  D=num_cell_dims(model)
  Triangulation(portion,ReferenceFE{D},model;kwargs...)
end

function Geometry.Triangulation(
  ::Type{ReferenceFE{D}}, model::DistributedDiscreteModel;kwargs...) where D
  Triangulation(no_ghost, ReferenceFE{D}, model; kwargs...)
end

function filter_cells_when_needed(
  portion::PArrays.WithGhost,
  cell_gids::AbstractIndexSet,
  trian::Triangulation)

  trian
end

function filter_cells_when_needed(
  portion::PArrays.NoGhost,
  cell_gids::AbstractIndexSet,
  trian::Triangulation)

  remove_ghost_cells(trian,cell_gids)
end

function filter_cells_when_needed(
  portion::FullyAssembledRows,
  cell_gids::AbstractIndexSet,
  trian::Triangulation)

  trian
end

function filter_cells_when_needed(
  portion::SubAssembledRows,
  cell_gids::AbstractIndexSet,
  trian::Triangulation)

  remove_ghost_cells(trian,cell_gids)
end

function remove_ghost_cells(trian::Triangulation,gids)
  model = get_background_model(trian)
   Dt    = num_cell_dims(model)
  #Dt    = num_cell_dims(trian)
  glue = get_glue(trian,Val(Dt))
  remove_ghost_cells(glue,trian,gids)
end

#function remove_ghost_cells(trian::Union{SkeletonTriangulation,BoundaryTriangulation},gids)
# model = get_background_model(trian)
# Dm    = num_cell_dims(model)
# glue = get_glue(trian,Val(Dm))
# remove_ghost_cells(glue,trian,gids)
#end

function remove_ghost_cells(glue::FaceToFaceGlue,trian,gids)
  tcell_to_mcell = glue.tface_to_mface
  mcell_to_part = gids.lid_to_part
  tcell_to_part = view(mcell_to_part,tcell_to_mcell)
  tcell_to_mask = tcell_to_part .== gids.part
  view(trian, findall(tcell_to_mask))
end

remove_ghost_cells(glue::Nothing,trian,gids) = trian

function remove_ghost_cells(glue::SkeletonPair,trian::SkeletonTriangulation,gids)
  ofacets = _find_owned_skeleton_facets(glue,gids)
  plus = view(trian.plus,ofacets)
  minus = view(trian.minus,ofacets)
  SkeletonTriangulation(plus,minus)
end

function remove_ghost_cells(glue::SkeletonPair,trian,gids)
  ofacets = _find_owned_skeleton_facets(glue,gids)
  view(trian,ofacets)
end

function _find_owned_skeleton_facets(glue,gids)
  glue_p = glue.plus
  glue_m = glue.minus
  T = eltype(gids.lid_to_part)
  ntfaces = length(glue_p.tface_to_mface)
  tface_to_part = zeros(T,ntfaces)
  for tface in 1:ntfaces
    mface_p = glue_p.tface_to_mface[tface]
    mface_m = glue_m.tface_to_mface[tface]
    gcell_p = gids.lid_to_gid[mface_p]
    gcell_m = gids.lid_to_gid[mface_m]
    if gcell_p > gcell_m
      part = gids.lid_to_part[mface_p]
    else
      part = gids.lid_to_part[mface_m]
    end
    tface_to_part[tface] = part
  end
  findall(part->part==gids.part,tface_to_part)
end

function add_ghost_cells(dtrian::DistributedTriangulation)
  dmodel = dtrian.model
  add_ghost_cells(dmodel,dtrian)
end

function _covers_all_faces(dmodel::DistributedDiscreteModel{Dm},
                           dtrian::DistributedTriangulation{Dt}) where {Dm,Dt}
  covers_all_faces=map_parts(dmodel.models,dtrian.trians) do model, trian
    glue = get_glue(trian,Val(Dt))
    @assert isa(glue,FaceToFaceGlue)
    isa(glue.tface_to_mface,IdentityVector)
  end
  reduce(&,covers_all_faces,init=true)
end

function add_ghost_cells(dmodel::DistributedDiscreteModel{Dm},
                         dtrian::DistributedTriangulation{Dt}) where {Dm,Dt}
  covers_all_faces=_covers_all_faces(dmodel,dtrian)
  if (covers_all_faces)
    trians = map_parts(dmodel.models) do model
      Triangulation(ReferenceFE{Dt},model)
    end
    DistributedTriangulation(trians,dmodel)
  else
    mcell_intrian = map_parts(dmodel.models,dtrian.trians) do model, trian
      glue = get_glue(trian,Val(Dt))
      @assert isa(glue,FaceToFaceGlue)
      nmcells = num_faces(model,Dt)
      mcell_intrian = fill(false,nmcells)
      tcell_to_mcell = glue.tface_to_mface
      mcell_intrian[tcell_to_mcell] .= true
      mcell_intrian
    end
    gids = get_face_gids(dmodel,Dt)
    exchange!(mcell_intrian,gids.exchanger)
    dreffes=map_parts(dmodel.models) do model
      ReferenceFE{Dt}
    end
    trians = map_parts(Triangulation,dreffes,dmodel.models,mcell_intrian)
    DistributedTriangulation(trians,dmodel)
  end
end

function generate_cell_gids(dtrian::DistributedTriangulation)
  dmodel = dtrian.model
  generate_cell_gids(dmodel,dtrian)
end

function generate_cell_gids(dmodel::DistributedDiscreteModel{Dm},
                            dtrian::DistributedTriangulation{Dt}) where {Dm,Dt}

  covers_all_faces=_covers_all_faces(dmodel,dtrian)
  if (covers_all_faces)
    get_face_gids(dmodel,Dt)
  else
    mgids = get_face_gids(dmodel,Dt)
    # count number owned cells
    notcells, tcell_to_mcell = map_parts(
      dmodel.models,dtrian.trians,mgids.partition) do model,trian,partition
      glue = get_glue(trian,Val(Dt))
      @assert isa(glue,FaceToFaceGlue)
      tcell_to_mcell = glue.tface_to_mface
      notcells = count(tcell_to_mcell) do mcell
        partition.lid_to_part[mcell] == partition.part
      end
      notcells, tcell_to_mcell
    end

    # Find the global range of owned dofs
    first_gtcell, ngtcellsplus1 = xscan(+,reduce,notcells,init=1)
    ngtcells = ngtcellsplus1 - 1

    # Assign global cell ids to owned cells
    mcell_to_gtcell = map_parts(
      first_gtcell,tcell_to_mcell,mgids.partition) do first_gtcell,tcell_to_mcell,partition
      mcell_to_gtcell = zeros(Int,length(partition.lid_to_part))
      gtcell = first_gtcell
      for mcell in tcell_to_mcell
        if partition.lid_to_part[mcell] == partition.part
          mcell_to_gtcell[mcell] = gtcell
          gtcell += 1
        end
      end
      mcell_to_gtcell
    end
    exchange!(mcell_to_gtcell,mgids.exchanger)

    # Prepare new partition
    partition = map_parts(mcell_to_gtcell,tcell_to_mcell,mgids.partition) do mcell_to_gtcell,tcell_to_mcell,partition
      tcell_to_gtcell = mcell_to_gtcell[tcell_to_mcell]
      tcell_to_part = partition.lid_to_part[tcell_to_mcell]
      IndexSet(partition.part,tcell_to_gtcell,tcell_to_part)
    end

    # Prepare the PRange
    neighbors = mgids.exchanger.parts_snd
    exchanger = Exchanger(partition,neighbors)
    gids = PRange(ngtcells,partition,exchanger)
    gids
  end
end

function _setup_face_gids!(dmodel::DistributedDiscreteModel{Dc},dim) where {Dc}
  Gridap.Helpers.@check 0 <= dim <= Dc
  if !isassigned(dmodel.face_gids,dim+1)
    mgids = dmodel.face_gids[Dc+1]
    nlfaces=map_parts(dmodel.models) do model
      num_faces(model,dim)
    end
    cell_lfaces=map_parts(dmodel.models) do model
      topo=get_grid_topology(model)
      faces=get_faces(topo, Dc, dim)
    end
    dmodel.face_gids[dim+1]=generate_gids(mgids,cell_lfaces,nlfaces)
  end
  return
end
