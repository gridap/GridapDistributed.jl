
# This might go to Gridap in the future. We keep it here for the moment.
function change_axes(a::Algebra.ArrayCounter,axes)
  @notimplemented
end

# This might go to Gridap in the future. We keep it here for the moment.
function change_axes(a::Algebra.CounterCOO{T,A}, axes::A) where {T,A}
  b=Algebra.CounterCOO{T}(axes)
  b.nnz = a.nnz
  b
end

# This might go to Gridap in the future. We keep it here for the moment.
function change_axes(a::Algebra.AllocationCOO{T,A}, axes::A) where {T,A}
  counter=change_axes(a.counter,axes)
  Algebra.AllocationCOO(counter,a.I,a.J,a.V)
end

function local_views(a::AbstractVector,rows)
  @notimplemented
end

function local_views(a::AbstractMatrix,rows,cols)
  @notimplemented
end

function consistent_local_views(a,ids,isconsistent)
  @abstractmethod
end

function local_views(a::AbstractArray)
  a
end

function local_views(a::PRange)
  partition(a)
end

function local_views(a::PVector)
  partition(a)
end

function local_views(a::PVector,rows::PRange)
  PArrays.local_view(a,rows)
end

function local_views(a::PSparseMatrix)
  partition(a)
end

function local_views(a::PSparseMatrix,rows::PRange,cols::PRange)
  PArrays.local_view(a,rows,cols)
end

function change_ghost(a::PVector,ids_fespace::PRange)
  if a.index_partition === partition(ids_fespace)
    a_fespace = a
  else
    a_fespace = similar(a,eltype(a),(ids_fespace,))
    a_fespace .= a
  end
  a_fespace
end

function consistent_local_views(a::PVector,ids_fespace::PRange,isconsistent)
  a_fespace = change_ghost(a,ids_fespace)
  if ! isconsistent
    assemble!((a,b)->b, partition(a_fespace),map(reverse,a_fespace.cache)) |> wait
  end
  partition(a_fespace)
end

function Algebra.allocate_vector(::Type{<:PVector{V,A}},ids::PRange) where {V,A}
  values = map(partition(ids)) do ids
    Tv = eltype(A)
    Tv(undef,length(local_to_owner(ids)))
  end
  PVector(values,partition(ids))
end

struct FullyAssembledRows end
struct SubAssembledRows end

# For the moment we use COO format even though
# it is quite memory consuming.
# We need to implement communication in other formats in
# PartitionedArrays.jl

struct PSparseMatrixBuilderCOO{T,B}
  local_matrix_type::Type{T}
  par_strategy::B
end

function Algebra.nz_counter(
  builder::PSparseMatrixBuilderCOO{A}, axs::Tuple{<:PRange,<:PRange}) where A
  rows, cols = axs
  counters = map(partition(rows),partition(cols)) do r,c
    axs = (Base.OneTo(local_length(r)),Base.OneTo(local_length(c)))
    Algebra.CounterCOO{A}(axs)
  end
  DistributedCounterCOO(builder.par_strategy,counters,rows,cols)
end

"""
"""
struct DistributedCounterCOO{A,B,C,D} <: DistributedGridapType
  par_strategy::A
  counters::B
  rows::C
  cols::D
  function DistributedCounterCOO(
    par_strategy,
    counters::AbstractArray{<:Algebra.CounterCOO},
    rows::PRange,
    cols::PRange)
    A = typeof(par_strategy)
    B = typeof(counters)
    C = typeof(rows)
    D = typeof(cols)
    new{A,B,C,D}(par_strategy,counters,rows,cols)
  end
end

function local_views(a::DistributedCounterCOO)
  a.counters
end

function local_views(a::DistributedCounterCOO,rows,cols)
  @check rows === a.rows
  @check cols === a.cols
  a.counters
end

function Algebra.nz_allocation(a::DistributedCounterCOO)
  allocs = map(nz_allocation,a.counters)
  DistributedAllocationCOO(a.par_strategy,allocs,a.rows,a.cols)
end

struct DistributedAllocationCOO{A,B,C,D} <:DistributedGridapType
  par_strategy::A
  allocs::B
  rows::C
  cols::D
  function DistributedAllocationCOO(
    par_strategy,
    allocs::AbstractArray{<:Algebra.AllocationCOO},
    rows::PRange,
    cols::PRange)
    A = typeof(par_strategy)
    B = typeof(allocs)
    C = typeof(rows)
    D = typeof(cols)
    new{A,B,C,D}(par_strategy,allocs,rows,cols)
  end
end

function change_axes(a::DistributedAllocationCOO{A,B,<:PRange,<:PRange},
                     axes::Tuple{<:PRange,<:PRange}) where {A,B}
  local_axes=map(partition(axes[1]),partition(axes[2])) do rows,cols
    (Base.OneTo(local_length(rows)), Base.OneTo(local_length(cols)))
  end
  allocs=map(change_axes,a.allocs,local_axes)
  DistributedAllocationCOO(a.par_strategy,allocs,axes[1],axes[2])
end

function local_views(a::DistributedAllocationCOO)
  a.allocs
end

function local_views(a::DistributedAllocationCOO,rows,cols)
  @check rows === a.rows
  @check cols === a.cols
  a.allocs
end

function first_gdof_from_ids(ids)
  lid_to_gid=local_to_global(ids) 
  owner_to_lid=own_to_local(ids)
  own_length(ids)>0 ? Int(lid_to_gid[first(owner_to_lid)]) : 1
end

function find_gid_and_owner(ighost_to_jghost,jindices)
  jghost_to_local=ghost_to_local(jindices)
  jlocal_to_global=local_to_global(jindices)
  jlocal_to_owner=local_to_owner(jindices)
  ighost_to_jlocal = view(jghost_to_local,ighost_to_jghost)

  ighost_to_global = jlocal_to_global[ighost_to_jlocal]
  ighost_to_owner = jlocal_to_owner[ighost_to_jlocal]
  ighost_to_global, ighost_to_owner
end

function Algebra.create_from_nz(a::PSparseMatrix)
  # For FullyAssembledRows the underlying Exchanger should
  # not have ghost layer making assemble! do nothing (TODO check)
  assemble!(a)
  a
end

function Algebra.create_from_nz(a::DistributedAllocationCOO{<:FullyAssembledRows})
  f(x) = nothing
  A, = _fa_create_from_nz_with_callback(f,a)
  A
end

# The given ids are assumed to be a sub-set of the lids
function ghost_lids_touched(a::AbstractLocalIndices,gids::AbstractVector{<:Integer})
  i = 0
  ghost_lids_touched = fill(false,ghost_length(a))
  glo_to_loc=global_to_local(a)
  loc_to_gho=local_to_ghost(a)
  for gid in gids
    lid = glo_to_loc[gid]
    ghost_lid = loc_to_gho[lid]
    if ghost_lid > 0 && !ghost_lids_touched[ghost_lid]
      ghost_lids_touched[ghost_lid] = true
      i += 1
    end
  end
  gids_ghost_lid_to_ghost_lid = Vector{Int32}(undef,i)
  i = 0
  ghost_lids_touched .= false
  for gid in gids
    lid = glo_to_loc[gid]
    ghost_lid = loc_to_gho[lid]
    if ghost_lid > 0 && !ghost_lids_touched[ghost_lid]
      ghost_lids_touched[ghost_lid] = true
      i += 1
      gids_ghost_lid_to_ghost_lid[i] = ghost_lid
    end
  end
  gids_ghost_lid_to_ghost_lid
end


function _fa_create_from_nz_with_callback(callback,a)

  # Recover some data
  I,J,C = map(a.allocs) do alloc
    alloc.I, alloc.J, alloc.V
  end
  parts = get_part_ids(a.allocs)
  rdofs = a.rows # dof ids of the test space
  cdofs = a.cols # dof ids of the trial space
  ngrdofs = length(rdofs)
  ngcdofs = length(cdofs)
  nordofs = map(own_length,partition(rdofs))
  nocdofs = map(own_length,partition(cdofs))
  first_grdof = map(first_gdof_from_ids,partition(rdofs))
  first_gcdof = map(first_gdof_from_ids,partition(cdofs))
  cneigs_snd = cdofs.exchanger.parts_snd
  cneigs_rcv = cdofs.exchanger.parts_rcv

  # This one has not ghost rows
  rows = PRange(
    parts,
    ngrdofs,
    nordofs,
    first_grdof)

  callback_output = callback(rows)

  # convert I and J to global dof ids
  to_global!(I,rdofs)
  to_global!(J,cdofs)

  # Find the ghost cols
  hcol_to_hcdof = ghost_lids_touched(cdofs,J)
  hcol_to_gid, hcol_to_part = map(
    find_gid_and_owner,hcol_to_hcdof,partition(cdofs))

  # Create the range for cols
  cols = PRange(
    parts,
    ngcdofs,
    nocdofs,
    first_gcdof,
    hcol_to_gid,
    hcol_to_part,
    cneigs_snd,
    cneigs_rcv)

  # Convert again I,J to local numeration
  to_lids!(I,rows)
  to_lids!(J,cols)

  # Adjust local matrix size to linear system's index sets
  b=change_axes(a,(rows,cols))

  # Compress local portions
  values = map(create_from_nz,b.allocs)

  # Build the matrix exchanger. This can be empty since no ghost rows
  exchanger = empty_exchanger(parts)

  # Finally build the matrix
  A = PSparseMatrix(values,rows,cols,exchanger)

  A, callback_output
end

function Algebra.create_from_nz(a::DistributedAllocationCOO{<:SubAssembledRows})
  f(x) = nothing
  A, = _sa_create_from_nz_with_callback(f,f,a)
  A
end

function _sa_create_from_nz_with_callback(callback,async_callback,a)
  # Recover some data
  I,J,V = map(a.allocs) do alloc
    alloc.I, alloc.J, alloc.V
  end |> tuple_of_arrays
  rdofs = a.rows # dof ids of the test space
  cdofs = a.cols # dof ids of the trial space
  ngrdofs = length(rdofs)
  ngcdofs = length(cdofs)

  # convert I and J to global dof ids
  map(to_global!,I,partition(rdofs))
  map(to_global!,J,partition(cdofs))

  # Find the ghost rows
  I_ghost_lids_to_rdofs_ghost_lids = map(ghost_lids_touched,partition(rdofs),I)
  I_ghost_to_global, I_ghost_to_owner = map(
    find_gid_and_owner,I_ghost_lids_to_rdofs_ghost_lids,partition(rdofs)) |> tuple_of_arrays

  rindices=map(partition(rdofs), 
               I_ghost_to_global, 
               I_ghost_to_owner) do rindices, ghost_to_global, ghost_to_owner 
     owner = part_id(rindices)
     own_indices=OwnIndices(ngrdofs,owner,own_to_global(rindices))
     ghost_indices=GhostIndices(ngrdofs,ghost_to_global,ghost_to_owner)
     OwnAndGhostIndices(own_indices,ghost_indices)
  end

  rows=PRange(rindices)

  # Move values to the owner part
  # since we have integrated only over owned cells
  t = PArrays.assemble_coo!(I,J,V,partition(rows))

  # Here we can overlap computations
  # This is a good place to overlap since
  # sending the matrix rows is a lot of data
  b = callback(rows)

  # Wait the transfer to finish
  wait(t)

  # Find the ghost cols
  J_ghost_lids_to_cdofs_ghost_lids = map(ghost_lids_touched,partition(cdofs),J)
  J_ghost_to_global, J_ghost_to_owner = map(
    find_gid_and_owner,J_ghost_lids_to_cdofs_ghost_lids,partition(cdofs)) |> tuple_of_arrays

  cindices=map(partition(cdofs), 
               J_ghost_to_global, 
               J_ghost_to_owner) do cindices, ghost_to_global, ghost_to_owner 
      owner = part_id(cindices)
      own_indices=OwnIndices(ngcdofs,owner,own_to_global(cindices))
      ghost_indices=GhostIndices(ngcdofs,ghost_to_global,ghost_to_owner)
      OwnAndGhostIndices(own_indices,ghost_indices)
  end

  # Create the range for cols
  cols = PRange(cindices)

  # Overlap rhs communications with CSC compression
  t2 = async_callback(b)

  # Convert again I,J to local numeration
  map(to_local!,I,partition(rows))
  map(to_local!,J,partition(cols))

  # Adjust local matrix size to linear system's index sets
  asys=change_axes(a,(rows,cols))

  # Compress the local matrices
  values = map(create_from_nz,asys.allocs)

  # Wait the transfer to finish
  if t2 !== nothing
    wait(t2)
  end

  # Finally build the matrix
  A = PSparseMatrix(values,partition(rows),partition(cols))

  A, b
end

struct PVectorBuilder{T,B}
  local_vector_type::Type{T}
  par_strategy::B
end

function Algebra.nz_counter(builder::PVectorBuilder,axs::Tuple{<:PRange})
  T = builder.local_vector_type
  rows, = axs
  counters = map(partition(rows)) do rows
    axs = (Base.OneTo(local_length(rows)),)
    nz_counter(ArrayBuilder(T),axs)
  end
  PVectorCounter(builder.par_strategy,counters,rows)
end

struct PVectorCounter{A,B,C}
  par_strategy::A
  counters::B
  rows::C
end

Algebra.LoopStyle(::Type{<:PVectorCounter}) = DoNotLoop()

function local_views(a::PVectorCounter)
  a.counters
end

function local_views(a::PVectorCounter,rows)
  @check rows === a.rows
  a.counters
end

function Arrays.nz_allocation(a::PVectorCounter{<:FullyAssembledRows})
  dofs = a.rows
  values = map(nz_allocation,a.counters)
  PVectorAllocationTrackOnlyValues(a.par_strategy,values,dofs)
end

struct PVectorAllocationTrackOnlyValues{A,B,C}
  par_strategy::A
  values::B
  rows::C
end

function local_views(a::PVectorAllocationTrackOnlyValues)
  a.values
end

function local_views(a::PVectorAllocationTrackOnlyValues,rows)
  @check rows === a.rows
  a.values
end

function Algebra.create_from_nz(a::PVectorAllocationTrackOnlyValues{<:FullyAssembledRows})
  # Create PRange for the rows of the linear system
  parts = get_part_ids(a.values)
  ngdofs = length(a.rows)
  nodofs = map(own_length,partition(a.rows))
  first_grdof = map(first_gdof_from_ids,partition(a.rows))

  # This one has no ghost rows
  rows = PRange(parts,ngdofs,nodofs,first_grdof)

  _rhs_callback(a,rows)
end

function Algebra.create_from_nz(a::PVectorAllocationTrackOnlyValues{<:SubAssembledRows})
  # This point MUST NEVER be reached. If reached there is an inconsistency
  # in the parallel code in charge of vector assembly
  @assert false
end

function _rhs_callback(c_fespace,rows)
  # The ghost values in b_fespace are aligned with the FESpace
  # but not with the ghost values in the rows of A
  b_fespace = PVector(c_fespace.values,partition(c_fespace.rows))

  # This one is aligned with the rows of A
  b = similar(b_fespace,eltype(b_fespace),(rows,))

  # First transfer owned values
  b .= b_fespace

  # Now transfer ghost
  function transfer_ghost(b,b_fespace,ids,ids_fespace)
    num_ghosts_vec = ghost_length(ids)
    gho_to_loc_vec = ghost_to_local(ids)
    loc_to_glo_vec = local_to_global(ids)
    gid_to_lid_fe  = global_to_local(ids_fespace)
    for ghost_lid_vec in 1:num_ghosts_vec
      lid_vec     = gho_to_loc_vec[ghost_lid_vec]
      gid         = loc_to_glo_vec[lid_vec]
      lid_fespace = gid_to_lid_fe[gid]
      b[lid_vec] = b_fespace[lid_fespace]
    end
  end
  map(
    transfer_ghost,
    partition(b),
    partition(b_fespace),
    b.index_partition,
    b_fespace.index_partition)

  return b
end

function Algebra.create_from_nz(a::PVector)
  # For FullyAssembledRows the underlying Exchanger should
  # not have ghost layer making assemble! do nothing (TODO check)
  assemble!(a)
  a
end

function Algebra.create_from_nz(
  a::DistributedAllocationCOO{<:FullyAssembledRows},
  c_fespace::PVectorAllocationTrackOnlyValues{<:FullyAssembledRows})

  function callback(rows)
    _rhs_callback(c_fespace,rows)
  end

  A,b = _fa_create_from_nz_with_callback(callback,a)
  A,b
end

struct PVectorAllocationTrackTouchedAndValues{A,B,C}
  allocations::A
  values::B
  rows::C
end

function Algebra.create_from_nz(
  a::DistributedAllocationCOO{<:SubAssembledRows},
  c_fespace::PVectorAllocationTrackOnlyValues{<:SubAssembledRows})

  function callback(rows)
    _rhs_callback(c_fespace,rows)
  end

  function async_callback(b)
    # now we can assemble contributions
    assemble!(b)
  end

  A,b = _sa_create_from_nz_with_callback(callback,async_callback,a)
  A,b
end

struct ArrayAllocationTrackTouchedAndValues{A}
  touched::Vector{Bool}
  values::A
end

Gridap.Algebra.LoopStyle(::Type{<:ArrayAllocationTrackTouchedAndValues}) = Gridap.Algebra.Loop()


function local_views(a::PVectorAllocationTrackTouchedAndValues,rows)
  @check rows === a.rows
  a.allocations
end

@inline function Arrays.add_entry!(c::Function,a::ArrayAllocationTrackTouchedAndValues,v,i,j)
  @notimplemented
end
@inline function Arrays.add_entry!(c::Function,a::ArrayAllocationTrackTouchedAndValues,v,i)
  if i>0
    if !(a.touched[i])
      a.touched[i]=true
    end
    if v!=nothing
      a.values[i]=c(v,a.values[i])
    end
  end
  nothing
end
@inline function Arrays.add_entries!(c::Function,a::ArrayAllocationTrackTouchedAndValues,v,i,j)
  @notimplemented
end
@inline function Arrays.add_entries!(c::Function,a::ArrayAllocationTrackTouchedAndValues,v,i)
  if v != nothing
    for (ve,ie) in zip(v,i)
      Arrays.add_entry!(c,a,ve,ie)
    end
  else
    for ie in i
      Arrays.add_entry!(c,a,nothing,ie)
    end
  end
  nothing
end

function Arrays.nz_allocation(a::DistributedCounterCOO{<:SubAssembledRows},
                              b::PVectorCounter{<:SubAssembledRows})
  A = nz_allocation(a)
  dofs = b.rows
  values = map(nz_allocation,b.counters)
  B=PVectorAllocationTrackOnlyValues(b.par_strategy,values,dofs)
  A,B
end

function Arrays.nz_allocation(a::PVectorCounter{<:SubAssembledRows})
  dofs = a.rows
  values = map(nz_allocation,a.counters)
  touched = map(values) do values
     fill!(Vector{Bool}(undef,length(values)),false)
  end
  allocations=map(values,touched) do values,touched
    ArrayAllocationTrackTouchedAndValues(touched,values)
  end
  PVectorAllocationTrackTouchedAndValues(allocations,values,dofs)
end

function local_views(a::PVectorAllocationTrackTouchedAndValues)
  a.allocations
end

function Algebra.create_from_nz(a::PVectorAllocationTrackTouchedAndValues)
   parts = get_part_ids(a.values)
   rdofs = a.rows # dof ids of the test space
   ngrdofs = length(rdofs)
   nordofs = map(own_length,partition(rdofs))
   first_grdof = map(first_gdof_from_ids,partition(rdofs))
   rneigs_snd = rdofs.exchanger.parts_snd
   rneigs_rcv = rdofs.exchanger.parts_rcv

   # Find the ghost rows
   hrow_to_hrdof=map(local_views(a.allocations),partition(rdofs)) do allocation, indices
    lids_touched=findall(allocation.touched)
    nhlids = count((x)->indices.lid_to_ohid[x]<0,lids_touched)
    hlids = Vector{Int32}(undef,nhlids)
    cur=1
    for lid in lids_touched
      hlid=indices.lid_to_ohid[lid]
      if hlid<0
        hlids[cur]=-hlid
        cur=cur+1
      end
    end
    hlids
   end
   hrow_to_gid, hrow_to_part = map(
       find_gid_and_owner,hrow_to_hrdof,partition(rdofs))

   # Create the range for rows
   rows = PRange(
           parts,
           ngrdofs,
           nordofs,
           first_grdof,
           hrow_to_gid,
           hrow_to_part,
           rneigs_snd,
           rneigs_rcv)

   b = _rhs_callback(a,rows)
   t2 = async_assemble!(b)

   # Wait the transfer to finish
   if t2 !== nothing
     map(schedule,t2)
     map(wait,t2)
   end
   b
end
