
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
struct DistributedGridTopology{Dc,Dp,A,B} <: GridapType
  topos::A
  face_gids::B
  function DistributedGridTopology(
    topos::AbstractArray{<:GridTopology{Dc,Dp}},
    face_gids::AbstractArray{<:PRange}
  ) where {Dc,Dp}
    A = typeof(topos)
    B = typeof(face_gids)
    new{Dc,Dp,A,B}(topos,face_gids)
  end
end

function DistributedGridTopology(
  topos::AbstractArray{<:GridTopology{Dc,Dp}}, cell_gids::PRange
) where {Dc,Dp}
  face_gids = Vector{PRange}(undef,Dc+1)
  face_gids[Dc+1] = cell_gids
  return DistributedGridTopology(topos,face_gids)
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

function get_cell_gids(topo::DistributedGridTopology{Dc}) where Dc
  topo.face_gids[Dc+1]
end

function get_face_gids(topo::DistributedGridTopology,dim::Integer)
  _setup_face_gids!(topo,dim)
  return topo.face_gids[dim+1]
end

function _setup_face_gids!(topo::DistributedGridTopology{Dc},dim) where {Dc}
  Gridap.Helpers.@check 0 <= dim <= Dc
  if !isassigned(topo.face_gids,dim+1)
    cell_gids = topo.face_gids[Dc+1]
    nlfaces = map(local_views(topo)) do topo
      num_faces(topo,dim)
    end
    cell_lfaces = map(local_views(topo)) do topo
      get_faces(topo, Dc, dim)
    end
    topo.face_gids[dim+1] = generate_gids(cell_gids,cell_lfaces,nlfaces)
  end
end

# In some cases, the orientation of locally computed faces is NOT consistent. 
# The following functions can be used to check for consistent orientation and fix it.
function _setup_consistent_faces!(topo::DistributedGridTopology)
  # Setting up consistent face-to-vertex maps should be enough
  # to guarantee consistent face orientation if it is done before 
  # any other face-to-face map is setup. So we should call this function 
  # just after creating the new models.
  D = num_cell_dims(topo)
  for dimfrom in 1:D-1
    _setup_consistent_faces!(topo, dimfrom, 0)
  end
end

function _setup_consistent_faces!(topo::DistributedGridTopology, dimfrom::Integer, dimto::Integer)
  @check 0 <= dimto <= dimfrom <= num_cell_dims(topo)
  gids_from = partition(get_face_gids(topo, dimfrom))
  gids_to   = partition(get_face_gids(topo, dimto))
  lfrom_to_gto = map(local_views(topo), gids_to) do topo, gids_to
    lfrom_to_lto = get_faces(topo, dimfrom, dimto)
    to_global!(lfrom_to_lto.data, gids_to)
    JaggedArray(lfrom_to_lto.data, lfrom_to_lto.ptrs)
  end
  wait(consistent!(PVector(lfrom_to_gto, gids_from)))
  map(lfrom_to_gto, gids_to) do lfrom_to_gto, gids_to
    to_local!(lfrom_to_gto.data, gids_to)
  end
  return nothing
end

function isconsistent_faces(topo::DistributedGridTopology)
  D = num_cell_dims(topo)
  for dimfrom in 1:D-1
    for dimto in 0:dimfrom-1
      !isconsistent_faces(topo, dimfrom, dimto) && return false
    end
  end
  return true
end

function isconsistent_faces(topo::DistributedGridTopology, dimfrom::Integer, dimto::Integer)
  @check 0 <= dimto <= dimfrom <= num_cell_dims(topo)
  gids_from = partition(get_face_gids(topo, dimfrom))
  gids_to   = partition(get_face_gids(topo, dimto))

  lfrom_to_lto = map(local_views(topo)) do topo
    get_faces(topo, dimfrom, dimto)
  end
  lfrom_to_gto = map(lfrom_to_lto, gids_to) do lfrom_to_lto, gids_to
    lto_gto = local_to_global(gids_to)
    JaggedArray(lto_gto[lfrom_to_lto.data],lfrom_to_lto.ptrs)
  end
  wait(consistent!(PVector(lfrom_to_gto, gids_from)))
  isconsistent = map(lfrom_to_lto, lfrom_to_gto, gids_to) do lfrom_to_lto, lfrom_to_gto, gids_to
    gto_to_lto = global_to_local(gids_to)
    lfrom_to_lto.data == gto_to_lto[lfrom_to_gto.data]
  end
  return reduce(&, isconsistent)
end

function Geometry.get_isboundary_face(topo::DistributedGridTopology, d::Integer)
  face_gids = get_face_gids(topo, d)
  is_local_boundary = map(local_views(topo)) do topo
    get_isboundary_face(topo,d)
  end
  t = assemble!(&,PVector(is_local_boundary, partition(face_gids)))
  is_global_boundary = partition(fetch(consistent!(fetch(t))))
  return is_global_boundary
end

"""
"""
struct DistributedFaceLabeling{A<:AbstractArray{<:FaceLabeling}}
  labels::A
end

local_views(a::DistributedFaceLabeling) = a.labels

function Geometry.add_tag_from_tags!(labels::DistributedFaceLabeling, name, tags)
  map(local_views(labels)) do labels
    add_tag_from_tags!(labels, name, tags)
  end
end

function Geometry.get_face_mask(labels::DistributedFaceLabeling, tags, d::Integer)
  map(local_views(labels)) do labels
    get_face_mask(labels, tags, d)
  end
end

function Geometry.FaceLabeling(topo::DistributedGridTopology)
  D = num_cell_dims(topo)
  labels = map(local_views(topo)) do topo
    d_to_ndfaces = [ num_faces(topo,d) for d in 0:D ]
    labels = FaceLabeling(d_to_ndfaces)
    for d in 0:D
      get_face_entity(labels,d) .= 1 # Interior as default
    end
    add_tag!(labels,"interior",[1])
    add_tag!(labels,"boundary",[2])
    return labels
  end
  for d in 0:D-1
    dface_to_is_boundary = get_isboundary_face(topo,d) # Global boundary
    map(labels,dface_to_is_boundary) do labels, dface_to_is_boundary
      dface_to_entity = get_face_entity(labels,d)
      dface_to_entity .+= dface_to_is_boundary
    end
  end
  return labels
end

# Distributed Discrete models
# We do not inherit from DiscreteModel on purpose.
# This object cannot implement the DiscreteModel interface in a strict sense

"""
"""
abstract type DistributedDiscreteModel{Dc,Dp} <: GridapType end

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
  DistributedGridTopology(map(get_grid_topology,local_views(model)),model.face_gids)
end

function Geometry.get_face_labeling(model::DistributedDiscreteModel)
  DistributedFaceLabeling(map(get_face_labeling,local_views(model)))
end

"""
"""
struct GenericDistributedDiscreteModel{Dc,Dp,A,B,C} <: DistributedDiscreteModel{Dc,Dp}
  models::A
  face_gids::B
  metadata::C
  function GenericDistributedDiscreteModel(
    models::AbstractArray{<:DiscreteModel{Dc,Dp}},
    face_gids::AbstractArray{<:PRange};
    metadata = nothing
  ) where {Dc,Dp}
    A = typeof(models)
    B = typeof(face_gids)
    C = typeof(metadata)
    new{Dc,Dp,A,B,C}(models,face_gids,metadata)
  end
end

function GenericDistributedDiscreteModel(
  models::AbstractArray{<:DiscreteModel{Dc,Dp}}, gids::PRange; metadata = nothing
) where {Dc,Dp}
  face_gids = Vector{PRange}(undef,Dc+1)
  face_gids[Dc+1] = gids
  GenericDistributedDiscreteModel(models,face_gids;metadata)
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
    cell_gids = dmodel.face_gids[Dc+1]
    nlfaces = map(local_views(dmodel)) do model
      num_faces(model,dim)
    end
    cell_lfaces = map(local_views(dmodel)) do model
      topo = get_grid_topology(model)
      get_faces(topo, Dc, dim)
    end
    dmodel.face_gids[dim+1] = generate_gids(cell_gids,cell_lfaces,nlfaces)
  end
end

# CartesianDiscreteModel
struct DistributedCartesianDescriptor{A,B,C,D}
  ranks::A
  mesh_partition::B
  descriptor::C
  ghost::D
  function DistributedCartesianDescriptor(
    ranks::AbstractArray{<:Integer},
    mesh_partition::NTuple{Dc,<:Integer},
    descriptor::CartesianDescriptor{Dc},
    ghost = map(i -> true, mesh_partition)
  ) where Dc
    A, B = typeof(ranks), typeof(mesh_partition)
    C, D = typeof(descriptor), typeof(ghost)
    new{A,B,C,D}(ranks,mesh_partition,descriptor,ghost)
  end
end

function Base.show(io::IO,k::MIME"text/plain",desc::DistributedCartesianDescriptor)
  ranks = desc.ranks
  map_main(ranks) do r
    nranks = desc.mesh_partition
    ncells = desc.descriptor.partition
    f(x) = join(x,"x")
    print(io,"$(f(ncells)) CartesianDescriptor distributed in $(f(nranks)) ranks")
  end
end

function emit_cartesian_descriptor(
  pdesc::Union{<:DistributedCartesianDescriptor{Dc},Nothing},
  new_ranks::AbstractArray{<:Integer},
  new_mesh_partition
) where Dc
  f(a) = Tuple(PartitionedArrays.getany(emit(a)))
  a, b, c, d, e = map(new_ranks) do rank
    if rank == 1
      desc = pdesc.descriptor
      @assert desc.map === identity
      Float64[desc.origin.data...], Float64[desc.sizes...], Int[desc.partition...], Bool[desc.isperiodic...], Int16[pdesc.ghost...]
    else
      Float64[], Float64[], Int[], Bool[], Int16[]
    end
  end |> tuple_of_arrays
  origin, sizes, partition, isperiodic, ghost = VectorValue(f(a)...), f(b), f(c), f(d), f(e)
  new_desc = CartesianDescriptor(origin,sizes,partition;isperiodic)
  return DistributedCartesianDescriptor(new_ranks,new_mesh_partition,new_desc,ghost)
end

const DistributedCartesianDiscreteModel{Dc,Dp,A,B,C} =
  GenericDistributedDiscreteModel{Dc,Dp,<:AbstractArray{<:CartesianDiscreteModel},B,<:DistributedCartesianDescriptor}

function Geometry.CartesianDiscreteModel(
  ranks::AbstractArray{<:Integer}, # Distributed array with the rank IDs
  parts::NTuple{N,<:Integer},      # Number of ranks (parts) in each direction
  args...; ghost = map(i -> true, parts), kwargs...
) where N
  desc = CartesianDescriptor(args...;kwargs...)
  @check N == length(desc.partition)
  @check prod(parts) == length(ranks)
  pdesc = DistributedCartesianDescriptor(ranks,parts,desc,ghost)
  return CartesianDiscreteModel(pdesc)
end

function Geometry.CartesianDiscreteModel(pdesc::DistributedCartesianDescriptor)
  desc = pdesc.descriptor
  isperiodic = desc.isperiodic
  if any(isperiodic)
    @notimplementedif pdesc.ghost != map(i->true,pdesc.mesh_partition)
    models, cell_indices = _cartesian_model_with_periodic_bcs(pdesc)
  else
    nc = desc.partition
    ranks = pdesc.ranks
    parts = pdesc.mesh_partition
    ghost = pdesc.ghost
    cell_indices = _uniform_partition(ranks,parts,nc,ghost,isperiodic)
    gcids  = CartesianIndices(nc)
    models = map(cell_indices) do cell_indices
      cmin = gcids[first(cell_indices)]
      cmax = gcids[last(cell_indices)]
      CartesianDiscreteModel(desc,cmin,cmax)
    end
  end
  gids = PRange(cell_indices)
  return GenericDistributedDiscreteModel(models,gids;metadata=pdesc)
end

function _cartesian_model_with_periodic_bcs(pdesc::DistributedCartesianDescriptor)
  ranks, parts, desc = pdesc.ranks, pdesc.mesh_partition, pdesc.descriptor

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

  return models, global_partition
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

  lcell_to_cell, lcell_to_part = map(parts) do part
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
    lcell_to_part = collect(Int32,view(cell_to_part,lcell_to_cell))
    lcell_to_cell, lcell_to_part
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
    Geometry.restrict(model,lcell_to_cell)
  end

  GenericDistributedDiscreteModel(models,gids)
end

# UnstructuredDiscreteModel

const DistributedUnstructuredDiscreteModel{Dc,Dp,A,B,C} =
  GenericDistributedDiscreteModel{Dc,Dp,<:AbstractArray{<:UnstructuredDiscreteModel},B,C}

function Geometry.UnstructuredDiscreteModel(model::GenericDistributedDiscreteModel)
  return GenericDistributedDiscreteModel(
    map(UnstructuredDiscreteModel,local_views(model)),
    get_cell_gids(model),
  )
end

# PolytopalDiscreteModel

function Geometry.PolytopalDiscreteModel(model::GenericDistributedDiscreteModel)
  pmodel = GenericDistributedDiscreteModel(
    map(Geometry.PolytopalDiscreteModel,local_views(model)),
    get_cell_gids(model)
  )
  _setup_consistent_faces!(get_grid_topology(pmodel))
  return pmodel
end

# Simplexify

function Geometry.simplexify(model::DistributedDiscreteModel;kwargs...)
  _model = UnstructuredDiscreteModel(model)
  ref_model = refine(_model; refinement_method = "simplexify", kwargs...)
  return UnstructuredDiscreteModel(Adaptivity.get_model(ref_model))
end

# Triangulation

# We do not inherit from Triangulation on purpose.
# This object cannot implement the Triangulation interface in a strict sense
"""
"""
struct DistributedTriangulation{Dc,Dp,A,B,C} <: GridapType
  trians  ::A
  model   ::B
  metadata::C
  function DistributedTriangulation(
    trians::AbstractArray{<:Triangulation{Dc,Dp}},
    model::DistributedDiscreteModel;
    metadata = nothing  
  ) where {Dc,Dp}
    A = typeof(trians)
    B = typeof(model)
    C = typeof(metadata)
    new{Dc,Dp,A,B,C}(trians,model,metadata)
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

function Geometry.num_cells(a::DistributedTriangulation{Df}) where Df
  model = get_background_model(a)
  gids = get_face_gids(model,Df)
  n_loc_ocells = map(local_views(a),partition(gids)) do a, gids
    glue = get_glue(a,Val(Df))
    @assert isa(glue,FaceToFaceGlue)
    tcell_to_mcell = glue.tface_to_mface
    if isa(tcell_to_mcell,IdentityVector)
      own_length(gids)
    else
      mcell_to_owned = local_to_own(gids)
      is_owned(mcell) = !iszero(mcell_to_owned[mcell])
      sum(is_owned,tcell_to_mcell;init=0)
    end
  end
  return sum(n_loc_ocells)
end

# Triangulation constructors

function Geometry.Triangulation(model::DistributedDiscreteModel;kwargs...)
  D = num_cell_dims(model)
  Triangulation(no_ghost,ReferenceFE{D},model;kwargs...)
end

function Geometry.Triangulation(::Type{ReferenceFE{D}},model::DistributedDiscreteModel;kwargs...) where D
  Triangulation(no_ghost, ReferenceFE{D}, model; kwargs...)
end

function Geometry.Triangulation(portion, model::DistributedDiscreteModel;kwargs...)
  D = num_cell_dims(model)
  Triangulation(portion,ReferenceFE{D},model;kwargs...)
end

function Geometry.Triangulation(
  portion,::Type{ReferenceFE{D}},model::DistributedDiscreteModel;kwargs...) where D
  gids = get_face_gids(model,D)
  trians = map(local_views(model)) do model
    Triangulation(ReferenceFE{D},model;kwargs...)
  end
  parent = DistributedTriangulation(trians,model)
  return filter_cells_when_needed(portion,gids,parent)
end

function Geometry.BoundaryTriangulation(model::DistributedDiscreteModel,args...;kwargs...)
  BoundaryTriangulation(no_ghost,model,args...;kwargs...)
end

function Geometry.BoundaryTriangulation(trian::DistributedTriangulation;kwargs...)
  BoundaryTriangulation(no_ghost,trian;kwargs...)
end

function Geometry.BoundaryTriangulation(
  portion,model::DistributedDiscreteModel;kwargs...)
  labels = get_face_labeling(model)
  Geometry.BoundaryTriangulation(portion,model,labels;kwargs...)
end

function Geometry.BoundaryTriangulation(
  portion,model::DistributedDiscreteModel,labels::DistributedFaceLabeling;tags=nothing)
  Dc = num_cell_dims(model)
  if isnothing(tags)
    topo = get_grid_topology(model)
    face_to_mask = get_isboundary_face(topo,Dc-1) # This is globally consistent
  else
    face_to_mask = get_face_mask(labels,tags,Dc-1)
  end
  Geometry.BoundaryTriangulation(portion,model,face_to_mask)
end

function Geometry.BoundaryTriangulation(
  portion,model::DistributedDiscreteModel,face_to_mask::AbstractArray)
  Dc = num_cell_dims(model)
  gids = get_face_gids(model,Dc)
  trians = map(local_views(model),face_to_mask) do model, face_to_mask
    BoundaryTriangulation(model,face_to_mask)
  end
  parent = DistributedTriangulation(trians,model)
  return filter_cells_when_needed(portion,gids,parent)
end

function Geometry.SkeletonTriangulation(model::DistributedDiscreteModel;kwargs...)
  SkeletonTriangulation(no_ghost,model;kwargs...)
end

function Geometry.SkeletonTriangulation(trian::DistributedTriangulation;kwargs...)
  SkeletonTriangulation(no_ghost,trian;kwargs...)
end

function Geometry.SkeletonTriangulation(
  portion,model::DistributedDiscreteModel;kwargs...)
  Dc = num_cell_dims(model)
  gids = get_face_gids(model,Dc)
  trians = map(local_views(model)) do model
    SkeletonTriangulation(model;kwargs...)
  end
  parent = DistributedTriangulation(trians,model)
  return filter_cells_when_needed(portion,gids,parent)
end

# NOTE: The following constructors require adding back the ghost cells:
# Potentially, the input `trian` has had some/all of its ghost cells removed. If we do not
# add them back, some skeleton facets might look like boundary facets to the local constructors...
function Geometry.BoundaryTriangulation(
  portion,trian::DistributedTriangulation;kwargs...
)
  model = get_background_model(trian)
  gids = get_cell_gids(model)
  ghosted_trian = add_ghost_cells(trian)
  trians = map(local_views(ghosted_trian)) do trian
    BoundaryTriangulation(trian;kwargs...)
  end
  parent = DistributedTriangulation(trians,model)
  return filter_cells_when_needed(portion,gids,parent)
end

function Geometry.SkeletonTriangulation(
  portion,trian::DistributedTriangulation;kwargs...
)
  model = get_background_model(trian)
  gids = get_cell_gids(model)
  ghosted_trian = add_ghost_cells(trian)
  trians = map(local_views(ghosted_trian)) do trian
    SkeletonTriangulation(trian;kwargs...)
  end
  parent = DistributedTriangulation(trians,model)
  return filter_cells_when_needed(portion,gids,parent)
end

function Geometry.Triangulation(portion,gids::AbstractLocalIndices, args...;kwargs...)
  trian = Triangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.BoundaryTriangulation(portion,gids::AbstractLocalIndices,args...;kwargs...)
  trian = BoundaryTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.SkeletonTriangulation(portion,gids::AbstractLocalIndices,args...;kwargs...)
  trian = SkeletonTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.InterfaceTriangulation(portion,gids::AbstractLocalIndices,args...;kwargs...)
  trian = InterfaceTriangulation(args...;kwargs...)
  filter_cells_when_needed(portion,gids,trian)
end

function Geometry.InterfaceTriangulation(a::DistributedTriangulation,b::DistributedTriangulation)
  @assert a.model === b.model
  trians = map(InterfaceTriangulation,a.trians,b.trians)
  DistributedTriangulation(trians,a.model)
end

# Filtering cells

@inline function filter_cells_when_needed(
  portion::Union{WithGhost,FullyAssembledRows},cell_gids,trian)
  return trian
end

@inline function filter_cells_when_needed(
  portion::Union{NoGhost,SubAssembledRows},cell_gids,trian)
  return remove_ghost_cells(trian,cell_gids)
end

# Removing ghost cells

struct RemoveGhostsMetadata{A}
  parents::A
end

function remove_ghost_cells(trian::DistributedTriangulation,gids)
  trians = map(remove_ghost_cells,local_views(trian),partition(gids))
  model  = get_background_model(trian)
  metadata = RemoveGhostsMetadata(local_views(trian))
  return DistributedTriangulation(trians,model;metadata)
end

function remove_ghost_cells(trian::Triangulation,gids)
  model = get_background_model(trian)
  Dt    = num_cell_dims(trian)
  glue  = get_glue(trian,Val(Dt))
  remove_ghost_cells(glue,trian,gids)
end

function remove_ghost_cells(
  trian::Union{SkeletonTriangulation,BoundaryTriangulation,Geometry.CompositeTriangulation},
  gids
)
  model = get_background_model(trian)
  Dm    = num_cell_dims(model)
  glue  = get_glue(trian,Val(Dm))
  remove_ghost_cells(glue,trian,gids)
end

function remove_ghost_cells(trian::AdaptedTriangulation,gids)
  AdaptedTriangulation(remove_ghost_cells(trian.trian,gids),trian.adapted_model)
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
  return findall(isequal(part_id(gids)),tface_to_part)
end

# Adding ghost cells

function add_ghost_cells(dtrian::DistributedTriangulation)
  dmodel = get_background_model(dtrian)
  add_ghost_cells(dmodel,dtrian)
end

function add_ghost_cells(dmodel::DistributedDiscreteModel,dtrian::DistributedTriangulation)
  add_ghost_cells(dtrian.metadata,dmodel,dtrian)
end

# We already have the parents saved up 
function add_ghost_cells(
  metadata::RemoveGhostsMetadata, dmodel::DistributedDiscreteModel{Dm}, dtrian::DistributedTriangulation{Dt}
) where {Dm,Dt}
  DistributedTriangulation(metadata.parents,dmodel)
end

# We have to reconstruct the ghosted triangulation
function add_ghost_cells(
  metadata, dmodel::DistributedDiscreteModel{Dm}, dtrian::DistributedTriangulation{Dt}
) where {Dm,Dt}

  tface_to_mface = map(local_views(dtrian)) do trian
    glue = get_glue(trian,Val(Dt))
    @assert isa(glue,FaceToFaceGlue)
    glue.tface_to_mface
  end

  # Case 1: All model faces are already in the triangulation
  covers_all_faces = reduce(&,map(x -> isa(x,IdentityVector), tface_to_mface),init=true)
  covers_all_faces && return dtrian

  # Otherwise: Add ghost cells to triangulation
  mface_intrian = map(local_views(dmodel),tface_to_mface) do model, tface_to_mface
    mface_intrian = fill(false,num_faces(model,Dt))
    mface_intrian[tface_to_mface] .= true
    mface_intrian
  end
  consistent!(PVector(mface_intrian,partition(get_face_gids(dmodel,Dt)))) |> wait

  # Case 2: New triangulation contains all model faces
  covers_all_faces = reduce(&,map(all,mface_intrian),init=true)
  covers_all_faces && return Triangulation(with_ghost,ReferenceFE{Dt},dmodel)

  # Case 3: Original triangulation already had the ghost cells
  new_tface_to_mface = map(findall,mface_intrian)
  had_ghost = reduce(&,map(==,tface_to_mface,new_tface_to_mface),init=true)
  had_ghost && return dtrian

  # Case 4: Otherwise, create a new triangulation with the ghost cells
  trians = map(local_views(dmodel),new_tface_to_mface) do model, new_tface_to_mface
    Triangulation(ReferenceFE{Dt},model,new_tface_to_mface)
  end
  return DistributedTriangulation(trians,dmodel)
end

function _covers_all_faces(
  dmodel::DistributedDiscreteModel{Dm},
  dtrian::DistributedTriangulation{Dt}
) where {Dm,Dt}
  covers_all_faces = map(local_views(dmodel),local_views(dtrian)) do model, trian
    glue = get_glue(trian,Val(Dt))
    @assert isa(glue,FaceToFaceGlue)
    isa(glue.tface_to_mface,IdentityVector)
  end
  reduce(&,covers_all_faces,init=true)
end

# Triangulation gids

function generate_cell_gids(dtrian::DistributedTriangulation)
  dmodel = get_background_model(dtrian)
  generate_cell_gids(dmodel,dtrian)
end

function generate_cell_gids(dmodel::DistributedDiscreteModel{Dm},
                            dtrian::DistributedTriangulation{Dt}) where {Dm,Dt}

  mgids = get_face_gids(dmodel,Dt)
  covers_all_faces = _covers_all_faces(dmodel,dtrian)
  if (covers_all_faces)
    tgids = mgids
  else
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
      first_gtcell,tcell_to_mcell,partition(mgids)) do first_gtcell,tcell_to_mcell,partition
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

    cache = fetch_vector_ghost_values_cache(mcell_to_gtcell,partition(mgids))
    fetch_vector_ghost_values!(mcell_to_gtcell,cache) |> wait

    # Prepare new partition
    ngtcells = reduction(+,notcells,destination=:all,init=zero(eltype(notcells)))
    indices = map(
      ngtcells,mcell_to_gtcell,tcell_to_mcell,partition(mgids)
    ) do ngtcells,mcell_to_gtcell,tcell_to_mcell,partition
      tcell_to_gtcell = mcell_to_gtcell[tcell_to_mcell]
      lid_to_owner  = local_to_owner(partition)
      tcell_to_part = lid_to_owner[tcell_to_mcell]
      LocalIndices(ngtcells,part_id(partition),tcell_to_gtcell,tcell_to_part)
    end
    _find_neighbours!(indices, partition(mgids))
    tgids = PRange(indices)
  end
  return tgids
end