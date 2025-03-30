
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
  if (a.index_partition === partition(ids))
    return a
  end
  a_new = change_ghost(T,a,ids)
  if make_consistent && !is_consistent
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

# This type is required in order to be able to access the local portion 
# of distributed sparse matrices and vectors using local indices from the 
# distributed test and trial spaces
struct LocalView{T,N,A} <:AbstractArray{T,N}
  plids_to_value::A
  d_to_lid_to_plid::NTuple{N,Vector{Int32}}
  local_size::NTuple{N,Int}
  function LocalView(
    plids_to_value::AbstractArray{T,N}, d_to_lid_to_plid::NTuple{N,Vector{Int32}}
  ) where {T,N}
    A = typeof(plids_to_value)
    local_size = map(length,d_to_lid_to_plid)
    new{T,N,A}(plids_to_value,d_to_lid_to_plid,local_size)
  end
end

Base.size(a::LocalView) = a.local_size
Base.IndexStyle(::Type{<:LocalView}) = IndexCartesian()
function Base.getindex(a::LocalView{T,N},lids::Vararg{Integer,N}) where {T,N}
  plids = map(getindex,a.d_to_lid_to_plid,lids)
  if all(i->i>0,plids)
    a.plids_to_value[plids...]
  else
    zero(T)
  end
end
function Base.setindex!(a::LocalView{T,N},v,lids::Vararg{Integer,N}) where {T,N}
  plids = map(getindex,a.d_to_lid_to_plid,lids)
  @check all(i->i>0,plids) "You are trying to set a value that is not stored in the local portion"
  a.plids_to_value[plids...] = v
end

function local_views(a::PVector,new_rows::PRange)
  old_rows = axes(a,1)
  if partition(old_rows) === partition(new_rows)
    partition(a)
  else
    map(partition(a),partition(old_rows),partition(new_rows)) do a,old_rows,new_rows
      LocalView(a,(local_to_local_map(new_rows,old_rows),))
    end
  end
end

function local_views(a::PSparseMatrix,new_rows::PRange,new_cols::PRange)
  old_rows, old_cols = axes(a)
  if (partition(old_rows) === partition(new_rows) && partition(old_cols) === partition(new_cols) )
    partition(a)
  else
    map(
      partition(a),partition(old_rows),partition(old_cols),partition(new_rows),partition(new_cols)
    ) do a,old_rows,old_cols,new_rows,new_cols
      rl2lmap = local_to_local_map(new_rows,old_rows)
      cl2lmap = local_to_local_map(new_cols,old_cols)
      LocalView(a,(rl2lmap,cl2lmap))
    end
  end
end

function local_views(a::BlockPVector,new_rows::BlockPRange)
  vals = map(local_views,blocks(a),blocks(new_rows)) |> to_parray_of_arrays
  return map(mortar,vals)
end

function local_views(a::BlockPMatrix,new_rows::BlockPRange,new_cols::BlockPRange)
  vals = map(CartesianIndices(blocksize(a))) do I
    local_views(blocks(a)[I],blocks(new_rows)[I],blocks(new_cols)[I])
  end |> to_parray_of_arrays
  return map(mortar,vals)
end
