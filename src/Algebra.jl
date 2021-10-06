
function local_views(a::AbstractVector)
  @abstractmethod
end

function consistent_local_views(a::AbstractVector)
  @abstractmethod
end

function local_views(a::AbstractPData)
  a
end

function local_views(a::PVector)
  a.values
end

function consistent_local_views(a::PVector)
  exchange!(a)
  a.values
end

function Albegra.allocate_vector(
  ::Type{T<:PVector},ids::PRange) where T
  T(undef,ids)
end

function local_views(a::PSparseMatrix)
  a.values
end

struct DistributedCounterCOO{A,B} <: GridapType
  counters::A
  gids::B
  function DistributedCounterCOO(
    counters::AbstractPData{<:CounterCOO},
    gids::PRange)
    A = typeof(counters)
    B = typeof(gids)
    new{A,B}(counters,gids)
  end
end

function Algebra.nz_counter(
  builder::SparseMatrixBuilder{<:PSparseMatrix{T,A}},
  axes::PRange) where {T,A}
  map_parts(axes.partition) do ids
    CounterCOO(eltype(A),Base.OneTo(num_lids(ids)))
  end
end

function nz_allocation(a::AbstractPData{<:CounterCOO})
  map_parts(nz_allocation,a)
end
