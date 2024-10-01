
# Vector allocation 

function Algebra.allocate_vector(::Type{<:PVector{V}},ids::PRange) where {V}
  PVector{V}(undef,partition(ids))
end

function Algebra.allocate_vector(::Type{<:BlockPVector{V}},ids::BlockPRange) where {V}
  BlockPVector{V}(undef,ids)
end

function Algebra.allocate_in_range(matrix::PSparseMatrix)
  V = Vector{eltype(matrix)}
  allocate_in_range(PVector{V},matrix)
end

function Algebra.allocate_in_domain(matrix::PSparseMatrix)
  V = Vector{eltype(matrix)}
  allocate_in_domain(PVector{V},matrix)
end

function Algebra.allocate_in_range(matrix::BlockPMatrix)
  V = Vector{eltype(matrix)}
  allocate_in_range(BlockPVector{V},matrix)
end

function Algebra.allocate_in_domain(matrix::BlockPMatrix)
  V = Vector{eltype(matrix)}
  allocate_in_domain(BlockPVector{V},matrix)
end

# PSparseMatrix copy

function Base.copy(a::PSparseMatrix)
  mats = map(copy,partition(a))
  cache = map(PartitionedArrays.copy_cache,a.cache)
  return PSparseMatrix(mats,partition(axes(a,1)),partition(axes(a,2)),cache)
end

# PartitionedArrays extras

function LinearAlgebra.axpy!(α,x::PVector,y::PVector)
  @check partition(axes(x,1)) === partition(axes(y,1))
  map(partition(x),partition(y)) do x,y
    LinearAlgebra.axpy!(α,x,y)
  end
  consistent!(y) |> wait
  return y
end

function LinearAlgebra.axpy!(α,x::BlockPVector,y::BlockPVector)
  map(blocks(x),blocks(y)) do x,y
    LinearAlgebra.axpy!(α,x,y)
  end
  return y
end

function Algebra.axpy_entries!(
  α::Number, A::PSparseMatrix, B::PSparseMatrix;
  check::Bool=true
)
# We should definitely check here that the index partitions are the same. 
# However: Because the different matrices are assembled separately, the objects are not the 
# same (i.e can't use ===). Checking the index partitions would then be costly...
  @assert reduce(&,map(PartitionedArrays.matching_local_indices,partition(axes(A,1)),partition(axes(B,1))))
  @assert reduce(&,map(PartitionedArrays.matching_local_indices,partition(axes(A,2)),partition(axes(B,2))))
  map(partition(A),partition(B)) do A, B
    Algebra.axpy_entries!(α,A,B;check)
  end
  return B
end

function Algebra.axpy_entries!(
  α::Number, A::BlockPMatrix, B::BlockPMatrix;
  check::Bool=true
)
  map(blocks(A),blocks(B)) do A, B
    Algebra.axpy_entries!(α,A,B;check)
  end
  return B
end

# This might go to Gridap in the future. We keep it here for the moment.
function change_axes(a::Algebra.ArrayCounter,axes)
  @notimplemented
end

# This might go to Gridap in the future. We keep it here for the moment.
function change_axes(a::Algebra.CounterCOO{T,A}, axes::A) where {T,A}
  b = Algebra.CounterCOO{T}(axes)
  b.nnz = a.nnz
  b
end

# This might go to Gridap in the future. We keep it here for the moment.
function change_axes(a::Algebra.AllocationCOO{T,A}, axes::A) where {T,A}
  counter = change_axes(a.counter,axes)
  Algebra.AllocationCOO(counter,a.I,a.J,a.V)
end

# Array of PArrays -> PArray of Arrays 
function to_parray_of_arrays(a::AbstractArray{<:MPIArray})
  indices = linear_indices(first(a))
  map(indices) do i
    map(a) do aj
      PartitionedArrays.getany(aj)
    end
  end
end

function to_parray_of_arrays(a::AbstractArray{<:DebugArray})
  indices = linear_indices(first(a))
  map(indices) do i
    map(a) do aj
      aj.items[i]
    end
  end
end

# This type is required because MPIArray from PArrays 
# cannot be instantiated with a NULL communicator
struct MPIVoidVector{T} <: AbstractVector{T}
  comm::MPI.Comm
  function MPIVoidVector(::Type{T}) where {T}
    new{T}(MPI.COMM_NULL)
  end
end

Base.size(a::MPIVoidVector) = (0,)
Base.IndexStyle(::Type{<:MPIVoidVector}) = IndexLinear()
function Base.getindex(a::MPIVoidVector,i::Int)
  error("Indexing of MPIVoidVector not possible.")
end
function Base.setindex!(a::MPIVoidVector,v,i::Int)
  error("Indexing of MPIVoidVector not possible.")
end
function Base.show(io::IO,k::MIME"text/plain",data::MPIVoidVector)
  println(io,"MPIVoidVector")
end

function num_parts(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    nparts = MPI.Comm_size(comm)
  else
    nparts = -1
  end
  nparts
end
@inline num_parts(comm::MPIArray) = num_parts(comm.comm)
@inline num_parts(comm::DebugArray) = length(comm.items)
@inline num_parts(comm::MPIVoidVector) = num_parts(comm.comm)

function get_part_id(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    id = MPI.Comm_rank(comm)+1
  else
    id = -1
  end
  id
end
@inline get_part_id(comm::MPIArray) = get_part_id(comm.comm)
@inline get_part_id(comm::MPIVoidVector) = get_part_id(comm.comm)

"""
    i_am_in(comm::MPIArray)
    i_am_in(comm::DebugArray)
  
  Returns `true` if the processor is part of the subcommunicator `comm`.
"""
function i_am_in(comm::MPI.Comm)
  get_part_id(comm) >=0
end
@inline i_am_in(comm::MPIArray) = i_am_in(comm.comm)
@inline i_am_in(comm::MPIVoidVector) = i_am_in(comm.comm)
@inline i_am_in(comm::DebugArray) = true

function change_parts(x::Union{MPIArray,DebugArray,Nothing,MPIVoidVector}, new_parts; default=nothing)
  x_new = map(new_parts) do p
    if isa(x,MPIArray)
      PartitionedArrays.getany(x)
    elseif isa(x,DebugArray) && (p <= length(x.items))
      x.items[p]
    else
      default
    end
  end
  return x_new
end

function generate_subparts(parts::MPIArray,new_comm_size)
  root_comm = parts.comm
  root_size = MPI.Comm_size(root_comm)
  rank = MPI.Comm_rank(root_comm)

  @static if isdefined(MPI,:MPI_UNDEFINED)
    mpi_undefined = MPI.MPI_UNDEFINED[]
  else
    mpi_undefined = MPI.API.MPI_UNDEFINED[]
  end
  
  if root_size == new_comm_size
    return parts
  else
    if rank < new_comm_size
      comm = MPI.Comm_split(root_comm,0,0)
      return distribute_with_mpi(LinearIndices((new_comm_size,));comm=comm,duplicate_comm=false)
    else
      comm = MPI.Comm_split(root_comm,mpi_undefined,mpi_undefined)
      return MPIVoidVector(eltype(parts))
    end
  end
end

function generate_subparts(parts::DebugArray,new_comm_size)
  DebugArray(LinearIndices((new_comm_size,)))
end

# local_views

function local_views(a)
  @abstractmethod
end

function get_parts(a)
  return linear_indices(local_views(a))
end

function local_views(a::AbstractVector,rows)
  @notimplemented
end

function local_views(a::AbstractMatrix,rows,cols)
  @notimplemented
end

local_views(a::AbstractArray) = a
local_views(a::PRange) = partition(a)
local_views(a::PVector) = partition(a)
local_views(a::PSparseMatrix) = partition(a)

function local_views(a::BlockPRange)
  map(blocks(a)) do a
    local_views(a)
  end |> to_parray_of_arrays
end

function local_views(a::BlockPArray)
  vals = map(blocks(a)) do a
    local_views(a)
  end |> to_parray_of_arrays
  return map(mortar,vals)
end

# change_ghost

function change_ghost(a::PVector{T},ids::PRange;is_consistent=false,make_consistent=false) where T
  same_partition = (a.index_partition === partition(ids))
  a_new = same_partition ? a : change_ghost(T,a,ids)
  if make_consistent && (!same_partition || !is_consistent)
    consistent!(a_new) |> wait
  end
  return a_new
end

function change_ghost(::Type{<:AbstractVector},a::PVector,ids::PRange)
  a_new = similar(a,eltype(a),(ids,))
  # Equivalent to copy!(a_new,a) but does not check that owned indices match
  map(copy!,own_values(a_new),own_values(a))
  return a_new
end

function change_ghost(::Type{<:OwnAndGhostVectors},a::PVector,ids::PRange)
  values = map(own_values(a),partition(ids)) do own_vals,ids
    ghost_vals = fill(zero(eltype(a)),ghost_length(ids))
    perm = PartitionedArrays.local_permutation(ids)
    OwnAndGhostVectors(own_vals,ghost_vals,perm)
  end
  return PVector(values,partition(ids))
end

function change_ghost(a::BlockPVector,ids::BlockPRange;is_consistent=false,make_consistent=false)
  vals = map(blocks(a),blocks(ids)) do a, ids
    change_ghost(a,ids;is_consistent=is_consistent,make_consistent=make_consistent)
  end
  return BlockPVector(vals,ids)
end

# This function computes a mapping among the local identifiers of a and b
# for which the corresponding global identifiers are both in a and b. 
# Note that the haskey check is necessary because in the general case 
# there might be gids in b which are not present in a
function find_local_to_local_map(a::AbstractLocalIndices,b::AbstractLocalIndices)
  a_local_to_b_local = fill(Int32(-1),local_length(a))
  b_local_to_global  = local_to_global(b)
  a_global_to_local  = global_to_local(a)
  for blid in 1:local_length(b)
    gid = b_local_to_global[blid]
    if a_global_to_local[gid] != zero(eltype(a_global_to_local))
      alid = a_global_to_local[gid]
      a_local_to_b_local[alid] = blid
    end  
  end
  a_local_to_b_local
end

# This type is required in order to be able to access the local portion 
# of distributed sparse matrices and vectors using local indices from the 
# distributed test and trial spaces
struct LocalView{T,N,A,B} <:AbstractArray{T,N}
  plids_to_value::A
  d_to_lid_to_plid::B
  local_size::NTuple{N,Int}
  function LocalView(
    plids_to_value::AbstractArray{T,N},d_to_lid_to_plid::NTuple{N}) where {T,N}
    A = typeof(plids_to_value)
    B = typeof(d_to_lid_to_plid)
    local_size = map(length,d_to_lid_to_plid)
    new{T,N,A,B}(plids_to_value,d_to_lid_to_plid,local_size)
  end
end

Base.size(a::LocalView) = a.local_size
Base.IndexStyle(::Type{<:LocalView}) = IndexCartesian()
function Base.getindex(a::LocalView{T,N},lids::Vararg{Integer,N}) where {T,N}
  plids = map(_lid_to_plid,lids,a.d_to_lid_to_plid)
  if all(i->i>0,plids)
    a.plids_to_value[plids...]
  else
    zero(T)
  end
end
function Base.setindex!(a::LocalView{T,N},v,lids::Vararg{Integer,N}) where {T,N}
  plids = map(_lid_to_plid,lids,a.d_to_lid_to_plid)
  @check all(i->i>0,plids) "You are trying to set a value that is not stored in the local portion"
  a.plids_to_value[plids...] = v
end

function _lid_to_plid(lid,lid_to_plid)
  plid = lid_to_plid[lid]
  plid
end

function local_views(a::PVector,new_rows::PRange)
  old_rows = axes(a,1)
  if partition(old_rows) === partition(new_rows)
    partition(a)
  else
    map(partition(a),partition(old_rows),partition(new_rows)) do vector_partition,old_rows,new_rows
      LocalView(vector_partition,(find_local_to_local_map(new_rows,old_rows),))
    end
  end
end

function local_views(a::PSparseMatrix,new_rows::PRange,new_cols::PRange)
  old_rows, old_cols = axes(a)
  if (partition(old_rows) === partition(new_rows) && partition(old_cols) === partition(new_cols) )
    partition(a)
  else
    map(partition(a),
        partition(old_rows),partition(old_cols),
        partition(new_rows),partition(new_cols)) do matrix_partition,old_rows,old_cols,new_rows,new_cols
      rl2lmap = find_local_to_local_map(new_rows,old_rows)
      cl2lmap = find_local_to_local_map(new_cols,old_cols)
      LocalView(matrix_partition,(rl2lmap,cl2lmap))
    end
  end
end

function local_views(a::BlockPVector,new_rows::BlockPRange)
  vals = map(local_views,blocks(a),blocks(new_rows)) |> to_parray_of_arrays
  return map(mortar,vals)
end

function local_views(a::BlockPMatrix,new_rows::BlockPRange,new_cols::BlockPRange)
  vals = map(CartesianIndices(blocksize(a))) do I
    local_views(blocks(a)[I],blocks(new_rows)[I[1]],blocks(new_cols)[I[2]])
  end |> to_parray_of_arrays
  return map(mortar,vals)
end

# PSparseMatrix assembly

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
  test_dofs_gids_prange, trial_dofs_gids_prange = axs
  counters = map(partition(test_dofs_gids_prange),partition(trial_dofs_gids_prange)) do r,c
    axs = (Base.OneTo(local_length(r)),Base.OneTo(local_length(c)))
    Algebra.CounterCOO{A}(axs)
  end
  DistributedCounterCOO(builder.par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
end

function Algebra.get_array_type(::PSparseMatrixBuilderCOO{Tv}) where Tv
  T = eltype(Tv)
  return PSparseMatrix{T}
end


"""
"""
struct DistributedCounterCOO{A,B,C,D} <: GridapType
  par_strategy::A
  counters::B
  test_dofs_gids_prange::C
  trial_dofs_gids_prange::D
  function DistributedCounterCOO(
    par_strategy,
    counters::AbstractArray{<:Algebra.CounterCOO},
    test_dofs_gids_prange::PRange,
    trial_dofs_gids_prange::PRange)
    A = typeof(par_strategy)
    B = typeof(counters)
    C = typeof(test_dofs_gids_prange)
    D = typeof(trial_dofs_gids_prange)
    new{A,B,C,D}(par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
  end
end

function local_views(a::DistributedCounterCOO)
  a.counters
end

function local_views(a::DistributedCounterCOO,test_dofs_gids_prange,trial_dofs_gids_prange)
  @check test_dofs_gids_prange === a.test_dofs_gids_prange
  @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
  a.counters
end

function Algebra.nz_allocation(a::DistributedCounterCOO)
  allocs = map(nz_allocation,a.counters)
  DistributedAllocationCOO(a.par_strategy,allocs,a.test_dofs_gids_prange,a.trial_dofs_gids_prange)
end

struct DistributedAllocationCOO{A,B,C,D} <: GridapType
  par_strategy::A
  allocs::B
  test_dofs_gids_prange::C
  trial_dofs_gids_prange::D
  function DistributedAllocationCOO(
    par_strategy,
    allocs::AbstractArray{<:Algebra.AllocationCOO},
    test_dofs_gids_prange::PRange,
    trial_dofs_gids_prange::PRange)
    A = typeof(par_strategy)
    B = typeof(allocs)
    C = typeof(test_dofs_gids_prange)
    D = typeof(trial_dofs_gids_prange)
    new{A,B,C,D}(par_strategy,allocs,test_dofs_gids_prange,trial_dofs_gids_prange)
  end
end

function change_axes(a::DistributedAllocationCOO{A,B,<:PRange,<:PRange},
                     axes::Tuple{<:PRange,<:PRange}) where {A,B}
  local_axes = map(partition(axes[1]),partition(axes[2])) do rows,cols
    (Base.OneTo(local_length(rows)), Base.OneTo(local_length(cols)))
  end
  allocs = map(change_axes,a.allocs,local_axes)
  DistributedAllocationCOO(a.par_strategy,allocs,axes[1],axes[2])
end

function change_axes(a::MatrixBlock{<:DistributedAllocationCOO},
                     axes::Tuple{<:Vector,<:Vector})
  block_ids  = CartesianIndices(a.array)
  rows, cols = axes
  array = map(block_ids) do I
    change_axes(a[I],(rows[I[1]],cols[I[2]]))
  end
  return ArrayBlock(array,a.touched)
end

function local_views(a::DistributedAllocationCOO)
  a.allocs
end

function local_views(a::DistributedAllocationCOO,test_dofs_gids_prange,trial_dofs_gids_prange)
  @check test_dofs_gids_prange === a.test_dofs_gids_prange
  @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
  a.allocs
end

function local_views(a::MatrixBlock{<:DistributedAllocationCOO})
  array = map(local_views,a.array) |> to_parray_of_arrays
  return map(ai -> ArrayBlock(ai,a.touched),array)
end

function get_allocations(a::DistributedAllocationCOO)
  I,J,V = map(local_views(a)) do alloc
    alloc.I, alloc.J, alloc.V
  end |> tuple_of_arrays
  return I,J,V
end

function get_allocations(a::ArrayBlock{<:DistributedAllocationCOO})
  tuple_of_array_of_parrays = map(get_allocations,a.array) |> tuple_of_arrays
  return tuple_of_array_of_parrays
end

get_test_gids(a::DistributedAllocationCOO)  = a.test_dofs_gids_prange
get_trial_gids(a::DistributedAllocationCOO) = a.trial_dofs_gids_prange
get_test_gids(a::ArrayBlock{<:DistributedAllocationCOO})  = map(get_test_gids,diag(a.array))
get_trial_gids(a::ArrayBlock{<:DistributedAllocationCOO}) = map(get_trial_gids,diag(a.array))

function Algebra.create_from_nz(a::PSparseMatrix)
  # For FullyAssembledRows the underlying Exchanger should
  # not have ghost layer making assemble! do nothing (TODO check)
  assemble!(a) |> wait
  a
end

function Algebra.create_from_nz(a::DistributedAllocationCOO{<:FullyAssembledRows})
  f(x) = nothing
  A, = _fa_create_from_nz_with_callback(f,a)
  return A
end

function Algebra.create_from_nz(a::ArrayBlock{<:DistributedAllocationCOO{<:FullyAssembledRows}})
  f(x) = nothing
  A, = _fa_create_from_nz_with_callback(f,a)
  return A
end

function _fa_create_from_nz_with_callback(callback,a)

  # Recover some data
  I,J,V = get_allocations(a)
  test_dofs_gids_prange  = get_test_gids(a)
  trial_dofs_gids_prange = get_trial_gids(a)

  rows = _setup_prange(test_dofs_gids_prange,I;ghost=false,ax=:rows)
  b = callback(rows)

  # convert I and J to global dof ids
  to_global_indices!(I,test_dofs_gids_prange;ax=:rows)
  to_global_indices!(J,trial_dofs_gids_prange;ax=:cols)

  # Create the range for cols
  cols = _setup_prange(trial_dofs_gids_prange,J;ax=:cols)

  # Convert again I,J to local numeration
  to_local_indices!(I,rows;ax=:rows)
  to_local_indices!(J,cols;ax=:cols)

  # Adjust local matrix size to linear system's index sets
  asys = change_axes(a,(rows,cols))

  # Compress local portions
  values = map(create_from_nz,local_views(asys))

  # Finally build the matrix
  A = _setup_matrix(values,rows,cols)
  return A, b
end

function Algebra.create_from_nz(a::DistributedAllocationCOO{<:SubAssembledRows})
  f(x) = nothing
  A, = _sa_create_from_nz_with_callback(f,f,a,nothing)
  return A
end

function Algebra.create_from_nz(a::ArrayBlock{<:DistributedAllocationCOO{<:SubAssembledRows}})
  f(x) = nothing
  A, = _sa_create_from_nz_with_callback(f,f,a,nothing)
  return A
end

function _sa_create_from_nz_with_callback(callback,async_callback,a,b)
  # Recover some data
  I,J,V = get_allocations(a)
  test_dofs_gids_prange = get_test_gids(a)
  trial_dofs_gids_prange = get_trial_gids(a)

  # convert I and J to global dof ids
  to_global_indices!(I,test_dofs_gids_prange;ax=:rows)
  to_global_indices!(J,trial_dofs_gids_prange;ax=:cols)

  # Create the Prange for the rows
  rows = _setup_prange(test_dofs_gids_prange,I;ax=:rows)
  
  # Move (I,J,V) triplets to the owner process of each row I.
  # The triplets are accompanyed which Jo which is the process column owner
  Jo = get_gid_owners(J,trial_dofs_gids_prange;ax=:cols)
  t  = _assemble_coo!(I,J,V,rows;owners=Jo)

  # Here we can overlap computations
  # This is a good place to overlap since
  # sending the matrix rows is a lot of data
  if !isa(b,Nothing)
    bprange=_setup_prange_from_pvector_allocation(b)
    b = callback(bprange)
  end

  # Wait the transfer to finish
  wait(t)

  # Create the Prange for the cols
  cols = _setup_prange(trial_dofs_gids_prange,J;ax=:cols,owners=Jo)

  # Overlap rhs communications with CSC compression
  t2 = async_callback(b)

  # Convert again I,J to local numeration
  to_local_indices!(I,rows;ax=:rows)
  to_local_indices!(J,cols;ax=:cols)

  # Adjust local matrix size to linear system's index sets
  asys = change_axes(a,(rows,cols))

  # Compress the local matrices
  values = map(create_from_nz,local_views(asys))

  # Wait the transfer to finish
  if !isa(t2,Nothing)
    wait(t2)
  end

  # Finally build the matrix
  A = _setup_matrix(values,rows,cols)
  return A, b
end


# PVector assembly 

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

function Algebra.get_array_type(::PVectorBuilder{Tv}) where Tv
  T = eltype(Tv)
  return PVector{T}
end

struct PVectorCounter{A,B,C}
  par_strategy::A
  counters::B
  test_dofs_gids_prange::C
end

Algebra.LoopStyle(::Type{<:PVectorCounter}) = DoNotLoop()

function local_views(a::PVectorCounter)
  a.counters
end

function local_views(a::PVectorCounter,rows)
  @check rows === a.test_dofs_gids_prange
  a.counters
end

function Arrays.nz_allocation(a::PVectorCounter{<:FullyAssembledRows})
  dofs = a.test_dofs_gids_prange
  values = map(nz_allocation,a.counters)
  PVectorAllocationTrackOnlyValues(a.par_strategy,values,dofs)
end

struct PVectorAllocationTrackOnlyValues{A,B,C}
  par_strategy::A
  values::B
  test_dofs_gids_prange::C
end

function local_views(a::PVectorAllocationTrackOnlyValues)
  a.values
end

function local_views(a::PVectorAllocationTrackOnlyValues,rows)
  @check rows === a.test_dofs_gids_prange
  a.values
end

function Algebra.create_from_nz(a::PVectorAllocationTrackOnlyValues{<:FullyAssembledRows})
  rows = _setup_prange_without_ghosts(a.test_dofs_gids_prange)
  _rhs_callback(a,rows)
end

function Algebra.create_from_nz(a::PVectorAllocationTrackOnlyValues{<:SubAssembledRows})
  # This point MUST NEVER be reached. If reached there is an inconsistency
  # in the parallel code in charge of vector assembly
  @assert false
end

function _rhs_callback(row_partitioned_vector_partition,rows)
  # The ghost values in row_partitioned_vector_partition are 
  # aligned with the FESpace but not with the ghost values in the rows of A
  b_fespace = PVector(row_partitioned_vector_partition.values,
                      partition(row_partitioned_vector_partition.test_dofs_gids_prange))

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
  assemble!(a) |> wait
  return a
end

function Algebra.create_from_nz(
  a::DistributedAllocationCOO{<:FullyAssembledRows},
  c_fespace::PVectorAllocationTrackOnlyValues{<:FullyAssembledRows})

  function callback(rows)
    _rhs_callback(c_fespace,rows)
  end

  A,b = _fa_create_from_nz_with_callback(callback,a)
  return A,b
end

struct PVectorAllocationTrackTouchedAndValues{A,B,C}
  allocations::A
  values::B
  test_dofs_gids_prange::C
end

function Algebra.create_from_nz(
  a::DistributedAllocationCOO{<:SubAssembledRows},
  b::PVectorAllocationTrackTouchedAndValues)

  function callback(rows)
    _rhs_callback(b,rows)
  end

  function async_callback(b)
    # now we can assemble contributions
    assemble!(b)
  end

  A,b = _sa_create_from_nz_with_callback(callback,async_callback,a,b)
  return A,b
end

struct ArrayAllocationTrackTouchedAndValues{A}
  touched::Vector{Bool}
  values::A
end

Gridap.Algebra.LoopStyle(::Type{<:ArrayAllocationTrackTouchedAndValues}) = Gridap.Algebra.Loop()


function local_views(a::PVectorAllocationTrackTouchedAndValues,rows)
  @check rows === a.test_dofs_gids_prange
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
    if !isa(v,Nothing)
      a.values[i]=c(v,a.values[i])
    end
  end
  nothing
end
@inline function Arrays.add_entries!(c::Function,a::ArrayAllocationTrackTouchedAndValues,v,i,j)
  @notimplemented
end
@inline function Arrays.add_entries!(c::Function,a::ArrayAllocationTrackTouchedAndValues,v,i)
  if !isa(v,Nothing)
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


function _setup_touched_and_allocations_arrays(values)
  touched = map(values) do values
    fill!(Vector{Bool}(undef,length(values)),false)
  end
  allocations = map(values,touched) do values,touched
   ArrayAllocationTrackTouchedAndValues(touched,values)
  end
  touched, allocations
end

function Arrays.nz_allocation(a::DistributedCounterCOO{<:SubAssembledRows},
                              b::PVectorCounter{<:SubAssembledRows})
  A      = nz_allocation(a)
  dofs   = b.test_dofs_gids_prange
  values = map(nz_allocation,b.counters)
  touched,allocations=_setup_touched_and_allocations_arrays(values)
  B = PVectorAllocationTrackTouchedAndValues(allocations,values,dofs)
  return A,B
end

function Arrays.nz_allocation(a::PVectorCounter{<:SubAssembledRows})
  dofs = a.test_dofs_gids_prange
  values = map(nz_allocation,a.counters)
  touched,allocations=_setup_touched_and_allocations_arrays(values)
  return PVectorAllocationTrackTouchedAndValues(allocations,values,dofs)
end

function local_views(a::PVectorAllocationTrackTouchedAndValues)
  a.allocations
end

function _setup_prange_from_pvector_allocation(a::PVectorAllocationTrackTouchedAndValues)

  # Find the ghost rows
  allocations = local_views(a.allocations)
  indices = partition(a.test_dofs_gids_prange)
  I_ghost_lids_to_dofs_ghost_lids = map(allocations, indices) do allocation, indices
    dofs_lids_touched = findall(allocation.touched)
    loc_to_gho = local_to_ghost(indices)
    n_I_ghost_lids = count((x)->loc_to_gho[x]!=0,dofs_lids_touched)
    I_ghost_lids = Vector{Int32}(undef,n_I_ghost_lids)
    cur = 1
    for lid in dofs_lids_touched
      dof_lid = loc_to_gho[lid]
      if dof_lid != 0
        I_ghost_lids[cur] = dof_lid
        cur = cur+1
      end
    end
    I_ghost_lids
  end

  ghost_to_global, ghost_to_owner = map(
    find_gid_and_owner,I_ghost_lids_to_dofs_ghost_lids,indices) |> tuple_of_arrays

  ngids = length(a.test_dofs_gids_prange)
  _setup_prange_impl_(ngids,indices,ghost_to_global,ghost_to_owner)
end

function Algebra.create_from_nz(a::PVectorAllocationTrackTouchedAndValues)
  rows = _setup_prange_from_pvector_allocation(a)
  b    = _rhs_callback(a,rows)
  t2   = assemble!(b)
  # Wait the transfer to finish
  if t2 !== nothing
    wait(t2)
  end
  return b
end

# Common Assembly Utilities
function first_gdof_from_ids(ids)
  if own_length(ids) == 0
    return 1
  end
  lid_to_gid   = local_to_global(ids) 
  owned_to_lid = own_to_local(ids)
  return Int(lid_to_gid[first(owned_to_lid)])
end

function find_gid_and_owner(ighost_to_jghost,jindices)
  jghost_to_local  = ghost_to_local(jindices)
  jlocal_to_global = local_to_global(jindices)
  jlocal_to_owner  = local_to_owner(jindices)
  ighost_to_jlocal = sort(view(jghost_to_local,ighost_to_jghost))

  ighost_to_global = jlocal_to_global[ighost_to_jlocal]
  ighost_to_owner  = jlocal_to_owner[ighost_to_jlocal]
  return ighost_to_global, ighost_to_owner
end

# The given ids are assumed to be a sub-set of the lids
function ghost_lids_touched(a::AbstractLocalIndices,gids::AbstractVector{<:Integer})
  glo_to_loc = global_to_local(a)
  loc_to_gho = local_to_ghost(a)
  
  # First pass: Allocate
  i = 0
  ghost_lids_touched = fill(false,ghost_length(a))
  for gid in gids
    lid = glo_to_loc[gid]
    ghost_lid = loc_to_gho[lid]
    if ghost_lid > 0 && !ghost_lids_touched[ghost_lid]
      ghost_lids_touched[ghost_lid] = true
      i += 1
    end
  end
  gids_ghost_lid_to_ghost_lid = Vector{Int32}(undef,i)

  # Second pass: fill 
  i = 1
  fill!(ghost_lids_touched,false)
  for gid in gids
    lid = glo_to_loc[gid]
    ghost_lid = loc_to_gho[lid]
    if ghost_lid > 0 && !ghost_lids_touched[ghost_lid]
      ghost_lids_touched[ghost_lid] = true
      gids_ghost_lid_to_ghost_lid[i] = ghost_lid
      i += 1
    end
  end

  return gids_ghost_lid_to_ghost_lid
end

# Find the neighbours of partition1 trying 
# to use those in partition2 as a hint 
function _find_neighbours!(partition1, partition2)
  partition2_snd, partition2_rcv = assembly_neighbors(partition2)
  partition2_graph = ExchangeGraph(partition2_snd, partition2_rcv)
  return assembly_neighbors(partition1; neighbors=partition2_graph)
end

# to_global! & to_local! analogs, needed for dispatching in block assembly

function to_local_indices!(I,ids::PRange;kwargs...)
  map(to_local!,I,partition(ids))
end

function to_global_indices!(I,ids::PRange;kwargs...)
  map(to_global!,I,partition(ids))
end

function get_gid_owners(I,ids::PRange;kwargs...)
  map(I,partition(ids)) do I, indices 
    glo_to_loc = global_to_local(indices) 
    loc_to_own = local_to_owner(indices)
    map(x->loc_to_own[glo_to_loc[x]], I)
  end 
end

for f in [:to_local_indices!, :to_global_indices!, :get_gid_owners]
  @eval begin
    function $f(I::Vector,ids::AbstractVector{<:PRange};kwargs...)
      map($f,I,ids)
    end

    function $f(I::Matrix,ids::AbstractVector{<:PRange};ax=:rows)
      @check ax ∈ [:rows,:cols]
      block_ids = CartesianIndices(I)
      map(block_ids) do id
        i = id[1]; j = id[2];
        if ax == :rows
          $f(I[i,j],ids[i])
        else
          $f(I[i,j],ids[j])
        end
      end
    end
  end
end

# _setup_matrix : local matrices + global PRanges -> Global matrix

function _setup_matrix(values,rows::PRange,cols::PRange)
  return PSparseMatrix(values,partition(rows),partition(cols))
end

function _setup_matrix(values,rows::Vector{<:PRange},cols::Vector{<:PRange})
  block_ids  = CartesianIndices((length(rows),length(cols)))
  block_mats = map(block_ids) do I
    block_values = map(v -> blocks(v)[I],values)
    return _setup_matrix(block_values,rows[I[1]],cols[I[2]])
  end
  return mortar(block_mats)
end

# _assemble_coo! : local coo triplets + global PRange -> Global coo values

function _assemble_coo!(I,J,V,rows::PRange;owners=nothing)
  if isa(owners,Nothing)
    PArrays.assemble_coo!(I,J,V,partition(rows))
  else
    assemble_coo_with_column_owner!(I,J,V,partition(rows),owners)
  end
end

function _assemble_coo!(I,J,V,rows::Vector{<:PRange};owners=nothing)
  block_ids = CartesianIndices(I)
  map(block_ids) do id
    i = id[1]; j = id[2];
    if isa(owners,Nothing)
      _assemble_coo!(I[i,j],J[i,j],V[i,j],rows[i])
    else
      _assemble_coo!(I[i,j],J[i,j],V[i,j],rows[i],owners=owners[i,j])
    end
  end
end

function assemble_coo_with_column_owner!(I,J,V,row_partition,Jown)
  """
    Returns three JaggedArrays with the coo triplets
    to be sent to the corresponding owner parts in parts_snd
  """
  function setup_snd(part,parts_snd,row_lids,coo_entries_with_column_owner)
    global_to_local_row = global_to_local(row_lids)
    local_row_to_owner = local_to_owner(row_lids)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    k_gi, k_gj, k_jo, k_v = coo_entries_with_column_owner
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    PArrays.length_to_ptrs!(ptrs)
    gi_snd_data = zeros(eltype(k_gi),ptrs[end]-1)
    gj_snd_data = zeros(eltype(k_gj),ptrs[end]-1)
    jo_snd_data = zeros(eltype(k_jo),ptrs[end]-1)
    v_snd_data = zeros(eltype(k_v),ptrs[end]-1)
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        gj = k_gj[k]
        v = k_v[k]
        p = ptrs[owner_to_i[owner]]
        gi_snd_data[p] = gi
        gj_snd_data[p] = gj
        jo_snd_data[p] = k_jo[k]
        v_snd_data[p]  = v
        k_v[k] = zero(v)
        ptrs[owner_to_i[owner]] += 1
      end
    end
    PArrays.rewind_ptrs!(ptrs)
    gi_snd = JaggedArray(gi_snd_data,ptrs)
    gj_snd = JaggedArray(gj_snd_data,ptrs)
    jo_snd = JaggedArray(jo_snd_data,ptrs)
    v_snd = JaggedArray(v_snd_data,ptrs)
    gi_snd, gj_snd, jo_snd, v_snd
  end
  """
    Pushes to coo_entries_with_column_owner the tuples 
    gi_rcv,gj_rcv,jo_rcv,v_rcv received from remote processes
  """
  function setup_rcv!(coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
    k_gi, k_gj, k_jo, k_v = coo_entries_with_column_owner
    current_n = length(k_gi)
    new_n = current_n + length(gi_rcv.data)
    resize!(k_gi,new_n)
    resize!(k_gj,new_n)
    resize!(k_jo,new_n)
    resize!(k_v,new_n)
    for p in 1:length(gi_rcv.data)
        k_gi[current_n+p] = gi_rcv.data[p]
        k_gj[current_n+p] = gj_rcv.data[p]
        k_jo[current_n+p] = jo_rcv.data[p]
        k_v[current_n+p] = v_rcv.data[p]
    end
  end
  part = linear_indices(row_partition)
  parts_snd, parts_rcv = assembly_neighbors(row_partition)
  coo_entries_with_column_owner = map(tuple,I,J,Jown,V)
  gi_snd, gj_snd, jo_snd, v_snd = map(setup_snd,part,parts_snd,row_partition,coo_entries_with_column_owner) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t1 = exchange(gi_snd,graph)
  t2 = exchange(gj_snd,graph)
  t3 = exchange(jo_snd,graph)
  t4 = exchange(v_snd,graph)
  @async begin
      gi_rcv = fetch(t1)
      gj_rcv = fetch(t2)
      jo_rcv = fetch(t3)
      v_rcv = fetch(t4)
      map(setup_rcv!,coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
      I,J,Jown,V
  end
end

# dofs_gids_prange can be either test_dofs_gids_prange or trial_dofs_gids_prange
# In the former case, gids is a vector of global test dof identifiers, while in the 
# latter, a vector of global trial dof identifiers
function _setup_prange(dofs_gids_prange::PRange,gids;ghost=true,owners=nothing,kwargs...)
  if !ghost
    _setup_prange_without_ghosts(dofs_gids_prange)
  elseif isa(owners,Nothing)
    _setup_prange_with_ghosts(dofs_gids_prange,gids)
  else
    _setup_prange_with_ghosts(dofs_gids_prange,gids,owners)
  end
end

function _setup_prange(
  dofs_gids_prange::AbstractVector{<:PRange},
  gids::AbstractMatrix;
  ax=:rows,ghost=true,owners=nothing
)
  @check ax ∈ (:rows,:cols)
  block_ids = LinearIndices(dofs_gids_prange)
  pvcat(x) = map(xi -> vcat(xi...), to_parray_of_arrays(x))

  gids_union, owners_union = map(block_ids,dofs_gids_prange) do id, prange
    gids_slice = (ax == :rows) ? gids[id,:] : gids[:,id]
    gids_union = pvcat(gids_slice)

    owners_union = nothing
    if !isnothing(owners)
      owners_slice = (ax == :rows) ? owners[id,:] : owners[:,id]
      owners_union = pvcat(owners_slice)
    end

    return gids_union, owners_union
  end |> tuple_of_arrays
  
  return map((p,g,o) -> _setup_prange(p,g;ghost=ghost,owners=o),dofs_gids_prange,gids_union,owners_union)
end

# Create PRange for the rows of the linear system
# without local ghost dofs as per required in the 
# FullyAssembledRows() parallel assembly strategy 
function _setup_prange_without_ghosts(dofs_gids_prange::PRange)
  ngdofs = length(dofs_gids_prange)
  indices = map(partition(dofs_gids_prange)) do dofs_indices 
    owner = part_id(dofs_indices)
    own_indices = OwnIndices(ngdofs,owner,own_to_global(dofs_indices))
    ghost_indices = GhostIndices(ngdofs,Int64[],Int32[])
    OwnAndGhostIndices(own_indices,ghost_indices)
  end
  return PRange(indices)
end

# Here we are assuming that the sparse communication graph underlying test_dofs_gids_partition
# is a superset of the one underlying indices. This is (has to be) true for the rows of the linear system.
# The precondition required for the consistency of any parallel assembly process in GridapDistributed 
# is that each processor can determine locally with a single layer of ghost cells the global indices and associated 
# processor owners of the rows that it touches after assembly of integration terms posed on locally-owned entities 
# (i.e., either cells or faces). 
function _setup_prange_with_ghosts(dofs_gids_prange::PRange,gids)
  ngdofs = length(dofs_gids_prange)
  dofs_gids_partition = partition(dofs_gids_prange)

  # Selected ghost ids -> dof PRange ghost ids
  gids_ghost_lids_to_dofs_ghost_lids = map(ghost_lids_touched,dofs_gids_partition,gids)

  # Selected ghost ids -> [global dof ids, owner processor ids]
  gids_ghost_to_global, gids_ghost_to_owner = map(
    find_gid_and_owner,gids_ghost_lids_to_dofs_ghost_lids,dofs_gids_partition) |> tuple_of_arrays

  return _setup_prange_impl_(ngdofs,dofs_gids_partition,gids_ghost_to_global,gids_ghost_to_owner)
end

# Here we cannot assume that the sparse communication graph underlying 
# trial_dofs_gids_partition is a superset of the one underlying indices.
# Here we chould check whether it is included and call _find_neighbours!()
# if this is the case. At present, we are not taking advantage of this, 
# but let the parallel scalable algorithm to compute the reciprocal to do the job. 
function _setup_prange_with_ghosts(dofs_gids_prange::PRange,gids,owners)
  ngdofs = length(dofs_gids_prange)
  dofs_gids_partition = partition(dofs_gids_prange)

  # Selected ghost ids -> [global dof ids, owner processor ids]
  gids_ghost_to_global, gids_ghost_to_owner = map(
    gids,owners,dofs_gids_partition) do gids, owners, indices
    ghost_touched   = Dict{Int,Bool}()
    ghost_to_global = Int64[] 
    ghost_to_owner  = Int64[]
    me = part_id(indices)
    for (j,jo) in zip(gids,owners)
      if jo != me
        if !haskey(ghost_touched,j)
          push!(ghost_to_global,j)
          push!(ghost_to_owner,jo)
          ghost_touched[j] = true
        end
      end
    end
    ghost_to_global, ghost_to_owner
  end |> tuple_of_arrays

  return _setup_prange_impl_(ngdofs,
                             dofs_gids_partition,
                             gids_ghost_to_global,
                             gids_ghost_to_owner;
                             discover_neighbours=false)
end 

function _setup_prange_impl_(ngdofs,
                             dofs_gids_partition,
                             gids_ghost_to_global,
                             gids_ghost_to_owner;
                             discover_neighbours=true)
  indices = map(dofs_gids_partition, 
                gids_ghost_to_global, 
                gids_ghost_to_owner) do dofs_indices, ghost_to_global, ghost_to_owner 
    owner = part_id(dofs_indices)
    own_indices   = OwnIndices(ngdofs,owner,own_to_global(dofs_indices))
    ghost_indices = GhostIndices(ngdofs,ghost_to_global,ghost_to_owner)
    OwnAndGhostIndices(own_indices,ghost_indices)
  end
  if discover_neighbours
    _find_neighbours!(indices,dofs_gids_partition)
  end
  return PRange(indices)
end
