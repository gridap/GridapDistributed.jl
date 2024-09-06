
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
