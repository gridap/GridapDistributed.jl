
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
    strategy :: Algebra.AssemblyStrategy
  ) where {T,N}
    A, B, S = typeof(counters), typeof(axes), typeof(strategy)
    new{S,T,N,A,B}(counters,axes,strategy)
  end
end

Base.axes(a::DistributedCounter) = a.axes
local_views(a::DistributedCounter) = a.counters

const PVectorCounter{S,T} = DistributedCounter{S,T<:Algebra.ArrayCounter,1}
Algebra.LoopStyle(::Type{<:PVectorCounter}) = DoNotLoop()

const PSparseMatrixCounter{S,T} = DistributedCounter{S,T<:Algebra.CounterCOO,2}
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
    strategy :: Algebra.AssemblyStrategy
  ) where {T,N}
    A, B, S = typeof(allocs), typeof(axes), typeof(strategy)
    new{S,T,N,A,B}(allocs,axes,strategy)
  end
end

Base.axes(a::DistributedAllocation) = a.axes
local_views(a::DistributedAllocation) = a.allocs

const PVectorAllocation{S,T} = DistributedAllocation{S,T,1}
const PSparseMatrixAllocation{S,T} = DistributedAllocation{S,T,2}

# PSparseMatrix assembly chain
#
#   1 - nz_counter(PSparseMatrixBuilder) -> PSparseMatrixCounter
#   2 - nz_allocation(PSparseMatrixCounter) -> PSparseMatrixAllocation
#   3 - create_from_nz(PSparseMatrixAllocation) -> PSparseMatrix

function Algebra.nz_counter(builder::PSparseMatrixBuilder,axs::NTuple{2,<:PRange})
  rows, cols = axs # test ids, trial ids
  counters = map(partition(rows),partition(cols)) do rows,cols
    axs = (Base.OneTo(local_length(rows)),Base.OneTo(local_length(cols)))
    Algebra.CounterCOO{A}(axs)
  end
  DistributedCounter(builder.par_strategy,counters,rows,cols)
end

function Algebra.nz_allocation(a::PSparseMatrixCounter)
  allocs = map(nz_allocation,local_views(a))
  DistributedAllocation(allocs,a.axes,a.strategy)
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
  rows = _setup_prange_without_ghosts(axes(a,1))
  _rhs_callback(a,rows)
end

function Algebra.create_from_nz(a::PVectorAllocation{S,<:TrackedArrayAllocation}) where S
  rows = _setup_prange_from_pvector_allocation(a)
  b    = _rhs_callback(a,rows)
  t2   = assemble!(b)
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
# This is done by using the following specializations chain:

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
    _rhs_callback(b,rows)
  end
  A, B = _fa_create_from_nz_with_callback(callback,a)
  return A, B
end

function Algebra.create_from_nz(
  a::PSparseMatrixAllocation{<:Assembled},
  b::PVectorAllocation{<:Assembled,<:TrackedArrayAllocation}
)
  function callback(rows)
    _rhs_callback(b,rows)
  end
  function async_callback(b)
    assemble!(b)
  end
  A, B = _sa_create_from_nz_with_callback(callback,async_callback,a,b)
  return A, B
end
