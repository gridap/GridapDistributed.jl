
# Parallel assembly strategies

struct Assembled <: FESpaces.AssemblyStrategy end
struct SubAssembled <: FESpaces.AssemblyStrategy end
struct LocallyAssembled <: FESpaces.AssemblyStrategy end

# PSparseMatrix and PVector builders

struct PSparseMatrixBuilder{T,B}
  local_matrix_type::Type{T}
  strategy::B
end

function Algebra.get_array_type(::PSparseMatrixBuilder{Tv}) where Tv
  T = eltype(Tv)
  return PSparseMatrix{T}
end

struct PVectorBuilder{T,B}
  local_vector_type::Type{T}
  strategy::B
end

function Algebra.get_array_type(::PVectorBuilder{Tv}) where Tv
  T = eltype(Tv)
  return PVector{T}
end

# Distributed counters and allocators

"""
    DistributedCounter{S,T,N} <: GridapType

Distributed N-dimensional counter, with local counters of type T.
Follows assembly strategy S.
"""
struct DistributedCounter{S,T,N,A,B} <: GridapType
  counters :: A
  axes     :: B
  strategy :: S
  function DistributedCounter(
    counters :: AbstractArray{T},
    axes     :: NTuple{N,<:PRange},
    strategy :: AssemblyStrategy
  ) where {T,N}
    A, B, S = typeof(counters), typeof(axes), typeof(strategy)
    new{S,T,N,A,B}(counters,axes,strategy)
  end
end

Base.axes(a::DistributedCounter) = a.axes
Base.axes(a::DistributedCounter,d::Integer) = a.axes[d]
local_views(a::DistributedCounter) = a.counters

const PVectorCounter{S,T,A,B} = DistributedCounter{S,T,1,A,B}
Algebra.LoopStyle(::Type{<:PVectorCounter}) = DoNotLoop()

const PSparseMatrixCounter{S,T,A,B} = DistributedCounter{S,T,2,A,B}
Algebra.LoopStyle(::Type{<:PSparseMatrixCounter}) = Loop()

"""
    DistributedAllocation{S,T,N} <: GridapType

Distributed N-dimensional allocator, with local allocators of type T. 
Follows assembly strategy S.
"""
struct DistributedAllocation{S,T,N,A,B} <: GridapType
  allocs   :: A
  axes     :: B
  strategy :: S
  function DistributedAllocation(
    allocs   :: AbstractArray{T},
    axes     :: NTuple{N,<:PRange},
    strategy :: AssemblyStrategy
  ) where {T,N}
    A, B, S = typeof(allocs), typeof(axes), typeof(strategy)
    new{S,T,N,A,B}(allocs,axes,strategy)
  end
end

Base.axes(a::DistributedAllocation) = a.axes
Base.axes(a::DistributedAllocation,d::Integer) = a.axes[d]
local_views(a::DistributedAllocation) = a.allocs

function change_axes(a::DistributedAllocation{S,T,N},axes::NTuple{N,<:PRange}) where {S,T,N}
  indices = map(partition,axes)
  local_axes = map(indices...) do indices...
    map(ids -> Base.OneTo(local_length(ids)), indices)
  end
  allocs = map(change_axes,a.allocs,local_axes)
  DistributedAllocation(allocs,axes,a.strategy)
end

const PVectorAllocation{S,T} = DistributedAllocation{S,T,1}
const PSparseMatrixAllocation{S,T} = DistributedAllocation{S,T,2}

function get_allocations(a::PSparseMatrixAllocation{S,<:Algebra.AllocationCOO}) where S
  I,J,V = map(local_views(a)) do alloc
    alloc.I, alloc.J, alloc.V
  end |> tuple_of_arrays
  return I,J,V
end

function collect_touched_ids(a::PVectorAllocation{<:TrackedArrayAllocation})
  map(local_views(a),partition(axes(a,1))) do a, ids
    rows = remove_ghost(unpermute(ids))

    ghost_lids = ghost_to_local(ids)
    ghost_owners = ghost_to_owner(ids)
    touched_ghost_lids = filter(lid -> a.touched[lid], ghost_lids)
    touched_ghost_owners = collect(ghost_owners[touched_ghost_lids])
    touched_ghost_gids = to_global!(touched_ghost_lids, ids)
    ghost = GhostIndices(n_global,touched_ghost_gids,touched_ghost_owners)
    return replace_ghost(rows,ghost)
  end
end

# PSparseMatrix assembly chain
#
#   1 - nz_counter(PSparseMatrixBuilder) -> PSparseMatrixCounter
#   2 - nz_allocation(PSparseMatrixCounter) -> PSparseMatrixAllocation
#   3 - create_from_nz(PSparseMatrixAllocation) -> PSparseMatrix

function Algebra.nz_counter(builder::PSparseMatrixBuilder{MT},axs::NTuple{2,<:PRange}) where MT
  rows, cols = axs # test ids, trial ids
  counters = map(partition(rows),partition(cols)) do rows, cols
    local_axs = (Base.OneTo(local_length(rows)),Base.OneTo(local_length(cols)))
    Algebra.CounterCOO{MT}(local_axs)
  end
  DistributedCounter(counters,axs,builder.strategy)
end

function Algebra.nz_allocation(a::PSparseMatrixCounter)
  allocs = map(nz_allocation,local_views(a))
  DistributedAllocation(allocs,a.axes,a.strategy)
end

function Algebra.create_from_nz(a::PSparseMatrix)
  assemble!(a) |> wait
  return a
end

function Algebra.create_from_nz(a::PSparseMatrixAllocation{<:LocallyAssembled})
  A, = create_from_nz_locally_assembled(a)
  return A
end

function Algebra.create_from_nz(a::PSparseMatrixAllocation{<:Assembled})
  A, = create_from_nz_assembled(a)
  return A
end

# PVector assembly chain:
#
#   1 - nz_counter(PVectorBuilder) -> PVectorCounter
#   2 - nz_allocation(PVectorCounter) -> PVectorAllocation
#   3 - create_from_nz(PVectorAllocation) -> PVector

function Algebra.nz_counter(builder::PVectorBuilder{VT},axs::Tuple{<:PRange}) where VT
  rows, = axs
  counters = map(partition(rows)) do rows
    axs = (Base.OneTo(local_length(rows)),)
    nz_counter(ArrayBuilder(VT),axs)
  end
  DistributedCounter(counters,axs,builder.strategy)
end

function Arrays.nz_allocation(a::PVectorCounter{<:LocallyAssembled})
  allocs = map(nz_allocation,local_views(a))
  DistributedAllocation(allocs,a.axes,a.strategy)
end

function Arrays.nz_allocation(a::PVectorCounter{<:Assembled})
  values = map(nz_allocation,local_views(a))
  allocs = map(TrackedArrayAllocation,values)
  return DistributedAllocation(allocs,a.axes,a.strategy)
end

function Algebra.create_from_nz(a::PVector)
  assemble!(a) |> wait
  return a
end

function Algebra.create_from_nz(a::PVectorAllocation{<:Assembled,<:AbstractVector})
  # This point MUST NEVER be reached. If reached there is an inconsistency
  # in the parallel code in charge of vector assembly
  @assert false
end

function Algebra.create_from_nz(a::PVectorAllocation{<:LocallyAssembled,<:AbstractVector})
  rows = remove_ghost(unpermute(axes(a,1)))
  values = local_views(a)
  b = rhs_callback(values,rows)
  return b
end

function Algebra.create_from_nz(a::PVectorAllocation{S,<:TrackedArrayAllocation}) where S
  rows = collect_touched_ids(a)
  values = map(ai -> ai.values, local_views(a))
  b  = rhs_callback(values,rows)
  t2 = assemble!(b)
  if t2 !== nothing
    wait(t2)
  end
  return b
end

# PSystem assembly chain:
#
# When assembling a full system (matrix + vector), it is more efficient to 
# overlap communications the assembly of the matrix and the vector.
# Not only it is faster, but also necessary to ensure identical ghost indices 
# in both the matrix and vector rows.
# This is done by using the following specializations:

function Arrays.nz_allocation(
  a::PSparseMatrixCounter{<:Assembled},
  b::PVectorCounter{<:Assembled}
)
  A = nz_allocation(a) # PSparseMatrixAllocation
  B = nz_allocation(b) # PVectorAllocation{<:Assembled,<:TrackedArrayAllocation}
  return A, B
end

function Algebra.create_from_nz(
  a::PSparseMatrixAllocation{<:LocallyAssembled},
  b::PVectorAllocation{<:LocallyAssembled,<:AbstractVector}
)
  function callback(rows)
    new_indices = partition(rows)
    values = local_views(b)
    b_fespace = PVector(values,partition(axes(b,1)))
    locally_repartition(b_fespace,new_indices)
  end
  A, B = create_from_nz_locally_assembled(a,callback)
  return A, B
end

function Algebra.create_from_nz(
  a::PSparseMatrixAllocation{<:Assembled},
  b::PVectorAllocation{<:Assembled,<:TrackedArrayAllocation}
)
  function callback(rows)
    new_indices = collect_touched_ids(b)
    values = map(ai -> ai.values, local_views(b))
    b_fespace = PVector(values,partition(axes(b,1)))
    locally_repartition(b_fespace,new_indices)
  end
  function async_callback(b)
    assemble!(b)
  end
  A, B = create_from_nz_assembled(a,callback,async_callback)
  return A, B
end

# Low-level assembly methods

function create_from_nz_locally_assembled(
  a,
  callback::Function = rows -> nothing
)
  I,J,V = get_allocations(a)
  test_ids = partition(axes(a,1))
  trial_ids = partition(axes(a,2))

  rows = map(remove_ghost,map(unpermute,test_ids))
  b = callback(rows)

  map(map_local_to_global!,I,test_ids)
  map(map_local_to_global!,J,trial_ids)

  # TODO: replace_ghost or union_ghost?
  # Actually, do we want to change the ghosts at all? We could return the unpermuted trial_ids
  J_owners = find_owner(trial_ids,J)
  cols = map(replace_ghost,map(unpermute,trial_ids),J,J_owners)

  map(map_global_to_local!,I,rows)
  map(map_global_to_local!,J,cols)

  assembled = true
  a_sys = change_axes(a,(PRange(rows),PRange(cols)))
  values = map(create_from_nz,local_views(a_sys))
  A = PSparseMatrix(values,rows,cols,assembled)

  return A, b
end

function create_from_nz_assembled(
  a,
  callback::Function = rows -> nothing,
  async_callback::Function = b -> empty_async_task
)
  I,J,V = get_allocations(a)
  test_ids = partition(axes(a,1))
  trial_ids = partition(axes(a,2))

  # convert I and J to global dof ids
  map(map_local_to_global!,I,test_ids)
  map(map_local_to_global!,J,trial_ids)

  # Overlapped COO communication and vector assembly
  rows = filter_and_replace_ghost(map(unpermute,test_ids),I)
  t = PartitionedArrays.assemble_coo!(I,J,V,rows)
  b = callback(rows)
  wait(t)

  # Overlap rhs communications with CSC compression
  t2 = async_callback(b)
  cols = filter_and_replace_ghost(map(unpermute,trial_ids),J)

  map(map_global_to_local!,I,rows)
  map(map_global_to_local!,J,cols)

  assembled = true
  a_sys = change_axes(a,(PRange(rows),PRange(cols)))
  values = map(create_from_nz,local_views(a_sys))
  A = PSparseMatrix(values,rows,cols,assembled)

  wait(t2)
  return A, b
end

function create_from_nz_subassembled(
  a,
  callback::Function = rows -> nothing,
  async_callback::Function = b -> empty_async_task
)
  rows = partition(axes(a,1))
  cols = partition(axes(a,2))

  b = callback(rows)
  t2 = async_callback(b)

  assembled = false
  values = map(create_from_nz,local_views(a))
  A = PSparseMatrix(values,rows,cols,assembled)

  wait(t2)
  return A, b
end