
struct WithGhost end
struct NoGhost end

const with_ghost = WithGhost()
const no_ghost = NoGhost()

# We do not inherit from Grid on purpose.
# This object cannot implement the Grid interface in a strict sense
"""
"""
struct DistributedGrid{Dc,Dp,A} <: GridapType
  grids::A
  function DistributedGrid(grids::AbstractArray{<:Grid{Dc,Dp}}) where {Dc,Dp}
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
  function DistributedGridTopology(topos::AbstractArray{<:GridTopology{Dc,Dp}}) where {Dc,Dp}
    A = typeof(topos)
    new{Dc,Dp,A}(topos)
  end
end

local_views(a::DistributedGridTopology) = a.topos

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
struct DistributedFaceLabeling{A<:AbstractArray{<:FaceLabeling}}
  labels::A
end

local_views(a::DistributedFaceLabeling) = a.labels

function Geometry.add_tag_from_tags!(labels::DistributedFaceLabeling, name, tags)
  map(labels.labels) do labels
    add_tag_from_tags!(labels, name, tags)
  end
end

# Dsitributed Discrete models
# We do not inherit from DiscreteModel on purpose.
# This object cannot implement the DiscreteModel interface in a strict sense

"""
"""
abstract type DistributedDiscreteModel{Dc,Dp} <: GridapType end

function generate_gids(::DistributedDiscreteModel)
  @abstractmethod
end

function get_cell_gids(model::DistributedDiscreteModel{Dc}) where Dc
  @abstractmethod
end

function get_face_gids(model::DistributedDiscreteModel,dim::Integer)
  @abstractmethod
end

Geometry.num_cell_dims(::DistributedDiscreteModel{Dc,Dp}) where {Dc,Dp} = Dc
Geometry.num_cell_dims(::Type{<:DistributedDiscreteModel{Dc,Dp}}) where {Dc,Dp} = Dc
Geometry.num_point_dims(::DistributedDiscreteModel{Dc,Dp}) where {Dc,Dp} = Dp
Geometry.num_point_dims(::Type{<:DistributedDiscreteModel{Dc,Dp}}) where {Dc,Dp} = Dp

function Geometry.num_cells(model::DistributedDiscreteModel{Dc}) where Dc
  length(get_cell_gids(model))
end

function Geometry.num_facets(model::DistributedDiscreteModel{Dc}) where Dc
  length(get_face_gids(model,Dc-1))
end

function Geometry.num_edges(model::DistributedDiscreteModel{Dc}) where Dc
  length(get_face_gids(model,1))
end

function Geometry.num_vertices(model::DistributedDiscreteModel{Dc}) where Dc
  length(get_face_gids(model,0))
end

function Geometry.num_faces(model::DistributedDiscreteModel{Dc},dim::Integer) where Dc
  length(get_face_gids(model,dim))
end

function Geometry.num_faces(model::DistributedDiscreteModel{Dc}) where Dc
  sum(d->num_faces(model,d),0:Dc)
end

function Geometry.get_grid(model::DistributedDiscreteModel)
  DistributedGrid(map(get_grid,local_views(model)))
end

function Geometry.get_grid_topology(model::DistributedDiscreteModel)
  DistributedGridTopology(map(get_grid_topology,local_views(model)))
end

function Geometry.get_face_labeling(model::DistributedDiscreteModel)
  DistributedFaceLabeling(map(get_face_labeling,local_views(model)))
end

"""
"""
struct GenericDistributedDiscreteModel{Dc,Dp,A,B} <: DistributedDiscreteModel{Dc,Dp}
  models::A
  face_gids::B
  function GenericDistributedDiscreteModel(
    models::AbstractArray{<:DiscreteModel{Dc,Dp}}, gids::PRange) where {Dc,Dp}
    A = typeof(models)
    face_gids=Vector{PRange}(undef,Dc+1)
    face_gids[Dc+1]=gids
    B = typeof(face_gids)
    new{Dc,Dp,A,B}(models,face_gids)
  end
end

# This is to support old API
function DistributedDiscreteModel(args...;kwargs...)
  GenericDistributedDiscreteModel(args...;kwargs...)
end

local_views(a::GenericDistributedDiscreteModel) = a.models

function get_cell_gids(model::GenericDistributedDiscreteModel{Dc}) where Dc
  model.face_gids[Dc+1]
end

function get_face_gids(model::GenericDistributedDiscreteModel,dim::Integer)
  _setup_face_gids!(model,dim)
  model.face_gids[dim+1]
end

function _setup_face_gids!(dmodel::GenericDistributedDiscreteModel{Dc},dim) where {Dc}
  Gridap.Helpers.@check 0 <= dim <= Dc
  if !isassigned(dmodel.face_gids,dim+1)
    mgids   = dmodel.face_gids[Dc+1]
    nlfaces = map(local_views(dmodel)) do model
      num_faces(model,dim)
    end
    cell_lfaces = map(local_views(dmodel)) do model
      topo  = get_grid_topology(model)
      faces = get_faces(topo, Dc, dim)
    end
    dmodel.face_gids[dim+1] = generate_gids(mgids,cell_lfaces,nlfaces)
  end
  return
end

# CartesianDiscreteModel
function Geometry.CartesianDiscreteModel(
  ranks::AbstractArray{<:Integer}, # Distributed array with the rank IDs
  parts::NTuple{N,<:Integer},      # Number of ranks (parts) in each direction
  args...;isperiodic=map(i->false,parts),kwargs...) where N 

  desc = CartesianDescriptor(args...;isperiodic=isperiodic,kwargs...)
  nc = desc.partition
  msg = """
    A CartesianDiscreteModel needs a Cartesian subdomain partition
    of the right dimensions.
  """
  @assert N == length(nc) msg

  if any(isperiodic)
    _cartesian_model_with_periodic_bcs(ranks,parts,desc)
  else
    ghost = map(i->true,parts)
    upartition = uniform_partition(ranks,parts,nc,ghost,isperiodic)
    gcids  = CartesianIndices(nc)
    models = map(ranks,upartition) do rank, upartition
      cmin = gcids[first(upartition)]
      cmax = gcids[last(upartition)]
      CartesianDiscreteModel(desc,cmin,cmax)  
    end
    gids = PRange(upartition)
    return GenericDistributedDiscreteModel(models,gids)
  end
end

function _cartesian_model_with_periodic_bcs(ranks,parts,desc)
  # We create and extended CartesianDescriptor for the local models: 
  # If a direction is periodic and partitioned: 
  #   - we add a ghost cell at either side, which will be made periodic by the index partition.
  #   - We move the origin to accomodate the new cells. 
  #   - We turn OFF the periodicity in the local model, since periodicity will be taken care of
  #     by the global index partition.
  _map = desc.map #! Important: the map should be periodic if you want to integrate on the ghost cells.
  _sizes  = desc.sizes
  _origin, _partition, _isperiodic = map(parts,desc.isperiodic,Tuple(desc.origin),_sizes,desc.partition) do np,isp,o,h,nc
    if isp && (np != 1)
      return o-h, nc+2, false
    else
      return o, nc, isp
    end
  end |> tuple_of_arrays
  _desc = CartesianDescriptor(Point(_origin),_sizes,_partition;map=_map,isperiodic=_isperiodic)

  # We create the global index partition, which has the original number of cells per direction. 
  # Globally, the periodicity is turned ON in the directions which are periodic and partitioned
  # (if a direction is not partitioned, the periodicity is handled locally).
  ghost = map(i->true,parts)
  global_isperiodic = map((isp,np) -> (np==1) ? false : isp, desc.isperiodic,parts)
  global_partition = uniform_partition(ranks,parts,desc.partition,ghost,global_isperiodic)

  # We create the local models:
  #  - We create the cartesian ranges for the extended partition, taking into account the periodicity
  #    in the directions that are periodic and partitioned.
  #  - We create the local models with the extended cells, and periodicity only in the directions
  #    that are periodic and NOT partitioned.
  ranges = map(ranks) do rank
    p = Tuple(CartesianIndices(parts)[rank])
    ranges = map(PartitionedArrays.local_range,p,parts,desc.partition,ghost,global_isperiodic)
    return map((r,isp,g,np) -> (isp && g && (np != 1)) ? r .+ 1 : r, ranges,global_isperiodic,ghost,parts)
  end
  cgids  = CartesianIndices(_partition)
  models = map(ranges) do range
    cmin = cgids[map(first,range)...]
    cmax = cgids[map(last,range)...]
    remove_boundary = map((p,n)->((p && (n!=1)) ? true : false),desc.isperiodic,parts)
    CartesianDiscreteModel(_desc,cmin,cmax,remove_boundary)
  end
  gids = PRange(global_partition)
  return GenericDistributedDiscreteModel(models,gids)
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
  parts::AbstractArray,
  model::DiscreteModel,
  cell_to_part::AbstractArray,
  cell_graph::SparseMatrixCSC = compute_cell_graph(model))

  ncells = num_cells(model)
  @assert length(cell_to_part) == ncells
  @assert size(cell_graph,1) == ncells
  @assert size(cell_graph,2) == ncells

  lcell_to_cell, lcell_to_part, gid_to_part = map(parts) do part
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
  end |> tuple_of_arrays

  partition = map(parts,lcell_to_cell,lcell_to_part) do part, lcell_to_cell, lcell_to_part
    LocalIndices(ncells, part, lcell_to_cell, lcell_to_part)
  end 

  # This is required to provide the hint that the communication 
  # pattern underlying partition is symmetric, so that we do not have 
  # to execute the algorithm the reconstructs the reciprocal in the 
  # communication graph
  assembly_neighbors(partition;symmetric=true)

  gids = PRange(partition)

  models = map(lcell_to_cell) do lcell_to_cell
    DiscreteModelPortion(model,lcell_to_cell)
  end

  GenericDistributedDiscreteModel(models,gids)
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
    trians::AbstractArray{<:Triangulation{Dc,Dp}},
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
  gids   = get_face_gids(model,Dt)
  trians = map(local_views(model),partition(gids)) do model, gids
    Triangulation(portion,gids,ReferenceFE{Dt},model;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

function Geometry.BoundaryTriangulation(
  portion,model::DistributedDiscreteModel{Dc};kwargs...) where Dc
  gids   = get_face_gids(model,Dc)
  trians = map(local_views(model),partition(gids)) do model, gids
    BoundaryTriangulation(portion,gids,model;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

function Geometry.SkeletonTriangulation(
  portion,model::DistributedDiscreteModel{Dc};kwargs...) where Dc
  gids   = get_face_gids(model,Dc)
  trians = map(local_views(model),partition(gids)) do model, gids
    SkeletonTriangulation(portion,gids,model;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

function Geometry.Triangulation(
  portion,gids::AbstractLocalIndices, args...;kwargs...)
  trian = Triangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.BoundaryTriangulation(
  portion,gids::AbstractLocalIndices,args...;kwargs...)
  trian = BoundaryTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.SkeletonTriangulation(
  portion,gids::AbstractLocalIndices,args...;kwargs...)
  trian = SkeletonTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.InterfaceTriangulation(
  portion,gids::AbstractLocalIndices,args...;kwargs...)
  trian = InterfaceTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.InterfaceTriangulation(a::DistributedTriangulation,b::DistributedTriangulation)
  trians = map(InterfaceTriangulation,a.trians,b.trians)
  @assert a.model === b.model
  DistributedTriangulation(trians,a.model)
end

function Geometry.Triangulation(
  portion, model::DistributedDiscreteModel;kwargs...)
  D = num_cell_dims(model)
  Triangulation(portion,ReferenceFE{D},model;kwargs...)
end

function Geometry.Triangulation(
  ::Type{ReferenceFE{D}}, model::DistributedDiscreteModel;kwargs...) where D
  Triangulation(no_ghost, ReferenceFE{D}, model; kwargs...)
end

function filter_cells_when_needed(
  portion::WithGhost,
  cell_gids::AbstractLocalIndices,
  trian::Triangulation)

  trian
end

function filter_cells_when_needed(
  portion::NoGhost,
  cell_gids::AbstractLocalIndices,
  trian::Triangulation)

  remove_ghost_cells(trian,cell_gids)
end

function filter_cells_when_needed(
  portion::FullyAssembledRows,
  cell_gids::AbstractLocalIndices,
  trian::Triangulation)

  trian
end

function filter_cells_when_needed(
  portion::SubAssembledRows,
  cell_gids::AbstractLocalIndices,
  trian::Triangulation)

  remove_ghost_cells(trian,cell_gids)
end

function remove_ghost_cells(trian::Triangulation,gids)
  model = get_background_model(trian)
  Dt    = num_cell_dims(trian)
  glue  = get_glue(trian,Val(Dt))
  remove_ghost_cells(glue,trian,gids)
end

function remove_ghost_cells(trian::Union{SkeletonTriangulation,BoundaryTriangulation},gids)
  model = get_background_model(trian)
  Dm    = num_cell_dims(model)
  glue  = get_glue(trian,Val(Dm))
  remove_ghost_cells(glue,trian,gids)
end

function remove_ghost_cells(glue::FaceToFaceGlue,trian,gids)
  tcell_to_mcell = glue.tface_to_mface
  mcell_to_part  = local_to_owner(gids)
  tcell_to_part  = view(mcell_to_part,tcell_to_mcell)
  tcell_to_mask  = tcell_to_part .== part_id(gids)
  view(trian, findall(tcell_to_mask))
end

function remove_ghost_cells(glue::SkeletonPair,trian::SkeletonTriangulation,gids)
  ofacets = _find_owned_skeleton_facets(glue,gids)
  plus    = view(trian.plus,ofacets)
  minus   = view(trian.minus,ofacets)
  SkeletonTriangulation(plus,minus)
end

function remove_ghost_cells(glue::SkeletonPair,trian,gids)
  ofacets = _find_owned_skeleton_facets(glue,gids)
  view(trian,ofacets)
end

function _find_owned_skeleton_facets(glue,gids)
  glue_p = glue.plus
  glue_m = glue.minus
  loc_to_own = local_to_owner(gids)
  loc_to_glo = local_to_global(gids)
  T = eltype(loc_to_own)
  ntfaces = length(glue_p.tface_to_mface)
  tface_to_part = zeros(T,ntfaces)
  for tface in 1:ntfaces
    mface_p = glue_p.tface_to_mface[tface]
    mface_m = glue_m.tface_to_mface[tface]
    gcell_p = loc_to_glo[mface_p]
    gcell_m = loc_to_glo[mface_m]
    if gcell_p > gcell_m
      part = loc_to_own[mface_p]
    else
      part = loc_to_own[mface_m]
    end
    tface_to_part[tface] = part
  end
  findall(part->part==part_id(gids),tface_to_part)
end

function add_ghost_cells(dtrian::DistributedTriangulation)
  dmodel = dtrian.model
  add_ghost_cells(dmodel,dtrian)
end

function _covers_all_faces(dmodel::DistributedDiscreteModel{Dm},
                           dtrian::DistributedTriangulation{Dt}) where {Dm,Dt}
  covers_all_faces=map(local_views(dmodel),local_views(dtrian)) do model, trian
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
    trians = map(local_views(dmodel)) do model
      Triangulation(ReferenceFE{Dt},model)
    end
    return DistributedTriangulation(trians,dmodel)
  else
    mcell_intrian = map(local_views(dmodel),local_views(dtrian)) do model, trian
      glue = get_glue(trian,Val(Dt))
      @assert isa(glue,FaceToFaceGlue)
      nmcells = num_faces(model,Dt)
      mcell_intrian = fill(false,nmcells)
      tcell_to_mcell = glue.tface_to_mface
      mcell_intrian[tcell_to_mcell] .= true
      mcell_intrian
    end
    gids = get_face_gids(dmodel,Dt)

    cache=fetch_vector_ghost_values_cache(mcell_intrian,partition(gids))
    fetch_vector_ghost_values!(mcell_intrian,cache) |> wait
    
    dreffes=map(local_views(dmodel)) do model
      ReferenceFE{Dt}
    end
    trians = map(Triangulation,dreffes,local_views(dmodel),mcell_intrian)
    return DistributedTriangulation(trians,dmodel)
  end
end

function generate_cell_gids(dtrian::DistributedTriangulation)
  dmodel = dtrian.model
  generate_cell_gids(dmodel,dtrian)
end

function generate_cell_gids(dmodel::DistributedDiscreteModel{Dm},
                            dtrian::DistributedTriangulation{Dt}) where {Dm,Dt}

  covers_all_faces = _covers_all_faces(dmodel,dtrian)
  if (covers_all_faces)
    get_face_gids(dmodel,Dt)
  else
    mgids = get_face_gids(dmodel,Dt)
    # count number owned cells
    notcells, tcell_to_mcell = map(
      local_views(dmodel),local_views(dtrian),PArrays.partition(mgids)) do model,trian,partition
      lid_to_owner = local_to_owner(partition)  
      part = part_id(partition)
      glue = get_glue(trian,Val(Dt))
      @assert isa(glue,FaceToFaceGlue)
      tcell_to_mcell = glue.tface_to_mface
      notcells = count(tcell_to_mcell) do mcell
        lid_to_owner[mcell] == part
      end
      notcells, tcell_to_mcell
    end |> tuple_of_arrays

    # Find the global range of owned dofs
    first_gtcell = scan(+,notcells,type=:exclusive,init=one(eltype(notcells)))

    # Assign global cell ids to owned cells
    mcell_to_gtcell = map(
      first_gtcell,tcell_to_mcell,PArrays.partition(mgids)) do first_gtcell,tcell_to_mcell,partition
      mcell_to_gtcell = zeros(Int,local_length(partition))
      loc_to_owner = local_to_owner(partition)
      part = part_id(partition)
      gtcell = first_gtcell
      for mcell in tcell_to_mcell
        if loc_to_owner[mcell] == part
          mcell_to_gtcell[mcell] = gtcell
          gtcell += 1
        end
      end
      mcell_to_gtcell
    end

    cache = fetch_vector_ghost_values_cache(mcell_to_gtcell,PArrays.partition(mgids))
    fetch_vector_ghost_values!(mcell_to_gtcell,cache) |> wait

    # Prepare new partition
    ngtcells = reduction(+,notcells,destination=:all,init=zero(eltype(notcells)))
    partition = map(ngtcells, 
                    mcell_to_gtcell,
                    tcell_to_mcell,
                    PArrays.partition(mgids)) do ngtcells,mcell_to_gtcell,tcell_to_mcell,partition
      tcell_to_gtcell = mcell_to_gtcell[tcell_to_mcell]
      lid_to_owner = local_to_owner(partition)
      tcell_to_part = lid_to_owner[tcell_to_mcell]
      LocalIndices(ngtcells,part_id(partition),tcell_to_gtcell,tcell_to_part)
    end
    _find_neighbours!(partition, PArrays.partition(mgids))
    gids = PRange(partition)
    gids
  end
end
