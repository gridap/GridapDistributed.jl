
"""
  struct BlockPRange{A} <: AbstractUnitRange{Int}
"""
struct BlockPRange{A} <: AbstractUnitRange{Int}
  ranges::Vector{PRange{A}}
  function BlockPRange(ranges::Vector{<:PRange{A}}) where A
    new{A}(ranges)
  end
end

Base.first(a::BlockPRange) = 1
Base.last(a::BlockPRange) = sum(map(last,a.ranges))

BlockArrays.blocklength(a::BlockPRange) = length(a.ranges)
BlockArrays.blocksize(a::BlockPRange) = (blocklength(a),)
BlockArrays.blockaxes(a::BlockPRange) = (Block.(Base.OneTo(blocklength(a))),)
BlockArrays.blocks(a::BlockPRange) = a.ranges

function PartitionedArrays.partition(a::BlockPRange)
  return map(partition,blocks(a)) |> to_parray_of_arrays
end

function Base.getindex(a::BlockPRange,inds::Block{1})
  a.ranges[inds.n...]
end

function PartitionedArrays.matching_local_indices(a::BlockPRange,b::BlockPRange)
  c = map(PartitionedArrays.matching_local_indices,blocks(a),blocks(b))
  reduce(&,c,init=true)
end

function PartitionedArrays.matching_own_indices(a::BlockPRange,b::BlockPRange)
  c = map(PartitionedArrays.matching_own_indices,blocks(a),blocks(b))
  reduce(&,c,init=true)
end

function PartitionedArrays.matching_ghost_indices(a::BlockPRange,b::BlockPRange)
  c = map(PartitionedArrays.matching_ghost_indices,blocks(a),blocks(b))
  reduce(&,c,init=true)
end

"""
  struct BlockPArray{V,T,N,A,B} <: BlockArrays.AbstractBlockArray{T,N}
"""
struct BlockPArray{V,T,N,A,B} <: BlockArrays.AbstractBlockArray{T,N}
  blocks::Array{A,N}
  axes::NTuple{N,B}

  function BlockPArray(blocks::Array{<:AbstractArray{T,N},N},
                       axes  ::NTuple{N,<:BlockPRange}) where {T,N}
    @check all(map(d->size(blocks,d)==blocklength(axes[d]),1:N))
    local_type(::Type{<:PVector{V}}) where V = V
    local_type(::Type{<:PSparseMatrix{V}}) where V = V
    A = eltype(blocks)
    B = typeof(first(axes))
    V = local_type(A)
    new{V,T,N,A,B}(blocks,axes)
  end
end

const BlockPVector{V,T,A,B} = BlockPArray{V,T,1,A,B}
const BlockPMatrix{V,T,A,B} = BlockPArray{V,T,2,A,B}

@inline function BlockPVector(blocks::Vector{<:PVector},rows::BlockPRange)
  BlockPArray(blocks,(rows,))
end

@inline function BlockPVector(blocks::Vector{<:PVector},rows::Vector{<:PRange})
  BlockPVector(blocks,BlockPRange(rows))
end

@inline function BlockPMatrix(blocks::Matrix{<:PSparseMatrix},rows::BlockPRange,cols::BlockPRange)
  BlockPArray(blocks,(rows,cols))
end

@inline function BlockPMatrix(blocks::Matrix{<:PSparseMatrix},rows::Vector{<:PRange},cols::Vector{<:PRange})
  BlockPMatrix(blocks,BlockPRange(rows),BlockPRange(cols))
end

function BlockPVector{V}(::UndefInitializer,rows::BlockPRange) where {V}
  vals = map(blocks(rows)) do r
    PVector{V}(undef,partition(r))
  end
  return BlockPVector(vals,rows)
end

function BlockPMatrix{V}(::UndefInitializer,rows::BlockPRange,cols::BlockPRange) where {V}
  block_ids  = CartesianIndices((blocklength(rows),blocklength(cols)))
  block_rows = blocks(rows)
  block_cols = blocks(cols)
  vals = map(block_ids) do I
    r = block_rows[I[1]]
    c = block_cols[I[2]]
    PSparseMatrix{V}(undef,partition(r),partition(c))
  end
  return BlockPMatrix(vals,rows)
end

# AbstractArray API

Base.axes(a::BlockPArray) = a.axes
Base.size(a::BlockPArray) = Tuple(map(length,a.axes))

Base.IndexStyle(::Type{<:BlockPVector}) = IndexLinear()
Base.IndexStyle(::Type{<:BlockPMatrix}) = IndexCartesian()

function Base.similar(a::BlockPVector,::Type{T},inds::Tuple{<:BlockPRange}) where T
  vals = map(blocks(a),blocks(inds[1])) do ai,i
    similar(ai,T,i)
  end
  return BlockPArray(vals,inds)
end

function Base.similar(::Type{<:BlockPVector{V,T,A}},inds::Tuple{<:BlockPRange}) where {V,T,A}
  rows   = blocks(inds[1])
  values = map(rows) do r
    return similar(A,(r,))
  end
  return BlockPArray(values,inds)
end

function Base.similar(a::BlockPMatrix,::Type{T},inds::Tuple{<:BlockPRange,<:BlockPRange}) where T
  vals = map(CartesianIndices(blocksize(a))) do I
    rows = inds[1].ranges[I[1]]
    cols = inds[2].ranges[I[2]]
    similar(a.blocks[I],T,(rows,cols))
  end
  return BlockPArray(vals,inds)
end

function Base.similar(::Type{<:BlockPMatrix{V,T,A}},inds::Tuple{<:BlockPRange,<:BlockPRange}) where {V,T,A}
  rows = blocks(inds[1])
  cols = blocks(inds[2])
  values = map(CartesianIndices((length(rows),length(cols)))) do I
    i,j = I[1],I[2]
    return similar(A,(rows[i],cols[j]))
  end
  return BlockPArray(values,inds)
end

function Base.getindex(a::BlockPArray{T,N},inds::Vararg{Int,N}) where {T,N}
  @error "Scalar indexing not supported"
end
function Base.setindex(a::BlockPArray{T,N},v,inds::Vararg{Int,N}) where {T,N}
  @error "Scalar indexing not supported"
end

function Base.show(io::IO,k::MIME"text/plain",data::BlockPArray{T,N}) where {T,N}
  v = first(blocks(data))
  s = prod(map(si->"$(si)x",blocksize(data)))[1:end-1]
  map_main(partition(v)) do values
      println(io,"$s-block BlockPArray{$T,$N}")
  end
end

function Base.zero(v::BlockPArray)
  return mortar(map(zero,blocks(v)))
end

function Base.copyto!(y::BlockPVector,x::BlockPVector)
  @check blocklength(x) == blocklength(y)
  yb, xb = blocks(y), blocks(x)
  for i in 1:blocksize(x,1)
    copyto!(yb[i],xb[i])
  end
  return y
end

function Base.copyto!(y::BlockPMatrix,x::BlockPMatrix)
  @check blocksize(x) == blocksize(y)
  yb, xb = blocks(y), blocks(x)
  for i in 1:blocksize(x,1)
    for j in 1:blocksize(x,2)
      copyto!(yb[i,j],xb[i,j])
    end
  end
  return y
end

function Base.fill!(a::BlockPVector,v)
  map(blocks(a)) do a
    fill!(a,v)
  end
  return a
end

function Base.sum(a::BlockPArray)
  # TODO: This could use a single communication, instead of one for each block
  # TODO: We could implement a generic reduce, that we apply to sum, all, any, etc..
  return sum(map(sum,blocks(a)))
end

Base.maximum(x::BlockPArray) = maximum(identity,x)
function Base.maximum(f::Function,x::BlockPArray)
  maximum(map(xi->maximum(f,xi),blocks(x)))
end

Base.minimum(x::BlockPArray) = minimum(identity,x)
function Base.minimum(f::Function,x::BlockPArray)
  minimum(map(xi->minimum(f,xi),blocks(x)))
end

function Base.:(==)(a::BlockPVector,b::BlockPVector)
  A = length(a) == length(b)
  B = all(map((ai,bi)->ai==bi,blocks(a),blocks(b)))
  return A && B
end

function Base.any(f::Function,x::BlockPVector)
  any(map(xi->any(f,xi),blocks(x)))
end

function Base.all(f::Function,x::BlockPVector)
  all(map(xi->all(f,xi),blocks(x)))
end

function LinearAlgebra.rmul!(a::BlockPVector,v::Number)
  map(ai->rmul!(ai,v),blocks(a))
  return a
end

# AbstractBlockArray API

BlockArrays.blocks(a::BlockPArray) = a.blocks

function Base.getindex(a::BlockPArray,inds::Block{1})
  a.blocks[inds.n...]
end
function Base.getindex(a::BlockPArray{V,T,N},inds::Block{N}) where {V,T,N}
  a.blocks[inds.n...]
end
function Base.getindex(a::BlockPArray{V,T,N},inds::Vararg{Block{1},N}) where {V,T,N}
  a.blocks[map(i->i.n[1],inds)...]
end

function BlockArrays.mortar(blocks::Vector{<:PVector})
  rows = map(b->axes(b,1),blocks)
  BlockPVector(blocks,rows)
end

function BlockArrays.mortar(blocks::Matrix{<:PSparseMatrix})
  rows = map(b->axes(b,1),blocks[:,1])
  cols = map(b->axes(b,2),blocks[1,:])

  function check_axes(a,r,c)
    A = PartitionedArrays.matching_local_indices(axes(a,1),r)
    B = PartitionedArrays.matching_local_indices(axes(a,2),c)
    return A & B
  end
  @check all(map(I -> check_axes(blocks[I],rows[I[1]],cols[I[2]]),CartesianIndices(size(blocks))))

  return BlockPMatrix(blocks,rows,cols)
end

# PartitionedArrays API

Base.wait(t::Array)  = map(wait,t)
Base.fetch(t::Array) = map(fetch,t)

function PartitionedArrays.assemble!(a::BlockPArray)
  map(assemble!,blocks(a))
end

function PartitionedArrays.consistent!(a::BlockPArray)
  map(consistent!,blocks(a))
end

function PartitionedArrays.partition(a::BlockPArray)
  vals = map(partition,blocks(a)) |> to_parray_of_arrays
  return map(mortar,vals)
end

function PartitionedArrays.to_trivial_partition(a::BlockPArray)
  vals = map(PartitionedArrays.to_trivial_partition,blocks(a))
  return mortar(vals)
end

function PartitionedArrays.local_values(a::BlockPArray)
  vals = map(local_values,blocks(a)) |> to_parray_of_arrays
  return map(mortar,vals)
end

function PartitionedArrays.own_values(a::BlockPArray)
  vals = map(own_values,blocks(a)) |> to_parray_of_arrays
  return map(mortar,vals)
end

function PartitionedArrays.ghost_values(a::BlockPArray)
  vals = map(ghost_values,blocks(a)) |> to_parray_of_arrays
  return map(mortar,vals)
end

function PartitionedArrays.own_ghost_values(a::BlockPMatrix)
  vals = map(own_ghost_values,blocks(a)) |> to_parray_of_arrays
  return map(mortar,vals)
end

function PartitionedArrays.ghost_own_values(a::BlockPMatrix)
  vals = map(ghost_own_values,blocks(a)) |> to_parray_of_arrays
  return map(mortar,vals)
end

# LinearAlgebra API

function Base.:*(a::Number,b::BlockPArray)
  mortar(map(bi -> a*bi,blocks(b)))
end
Base.:*(b::BlockPMatrix,a::Number) = a*b
Base.:/(b::BlockPVector,a::Number) = (1/a)*b

function Base.:*(a::BlockPMatrix,b::BlockPVector)
  c = similar(b)
  mul!(c,a,b)
  return c
end

for op in (:+,:-)
  @eval begin
    function Base.$op(a::BlockPArray)
      mortar(map($op,blocks(a)))
    end
    function Base.$op(a::BlockPArray,b::BlockPArray)
      @assert blocksize(a) == blocksize(b)
      mortar(map($op,blocks(a),blocks(b)))
    end
  end
end

function LinearAlgebra.mul!(y::BlockPVector,A::BlockPMatrix,x::BlockPVector)
  o = one(eltype(A))
  mul!(y,A,x,o,o)
end

function LinearAlgebra.mul!(y::BlockPVector,A::BlockPMatrix,x::BlockPVector,α::Number,β::Number)
  yb, Ab, xb = blocks(y), blocks(A), blocks(x)
  z = zero(eltype(y))
  o = one(eltype(A))
  for i in 1:blocksize(A,1)
    fill!(yb[i],z)
    for j in 1:blocksize(A,2)
      mul!(yb[i],Ab[i,j],xb[j],α,o)
    end
    rmul!(yb[i],β)
  end
  return y
end

function LinearAlgebra.dot(x::BlockPVector,y::BlockPVector)
  return sum(map(dot,blocks(x),blocks(y)))
end

function LinearAlgebra.norm(v::BlockPVector,p::Real=2)
  if p == 2
    # More accurate, I think, given the fact we are not 
    # repeating the sqrt(square(sqrt...)) process in every block and every processor
    return sqrt(dot(v,v)) 
  end
  block_norms = map(vi->norm(vi,p),blocks(v))
  return sum(block_norms.^p)^(1/p)
end

function LinearAlgebra.fillstored!(a::BlockPMatrix,v)
  map(blocks(a)) do a
    LinearAlgebra.fillstored!(a,v)
  end
  return a
end

# Broadcasting

struct BlockPBroadcasted{A,B}
  blocks :: A
  axes   :: B
end

BlockArrays.blocks(b::BlockPBroadcasted) = b.blocks
BlockArrays.blockaxes(b::BlockPBroadcasted) = b.axes

function Base.broadcasted(f, args::Union{BlockPArray,BlockPBroadcasted}...)
  a1 = first(args)
  @boundscheck @assert all(ai -> blockaxes(ai) == blockaxes(a1),args)
  
  blocks_in = map(blocks,args)
  blocks_out = map((largs...)->Base.broadcasted(f,largs...),blocks_in...)
  
  return BlockPBroadcasted(blocks_out,blockaxes(a1))
end

function Base.broadcasted(f, a::Number, b::Union{BlockPArray,BlockPBroadcasted})
  blocks_out = map(b->Base.broadcasted(f,a,b),blocks(b))
  return BlockPBroadcasted(blocks_out,blockaxes(b))
end

function Base.broadcasted(f, a::Union{BlockPArray,BlockPBroadcasted}, b::Number)
  blocks_out = map(a->Base.broadcasted(f,a,b),blocks(a))
  return BlockPBroadcasted(blocks_out,blockaxes(a))
end

function Base.broadcasted(
  f,
  a::Union{BlockPArray,BlockPBroadcasted},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}
)
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  b::Union{BlockPArray,BlockPBroadcasted}
)
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::BlockPBroadcasted)
  blocks_out = map(Base.materialize,blocks(b))
  return mortar(blocks_out)
end

function Base.materialize!(a::BlockPArray,b::BlockPBroadcasted)
  map(Base.materialize!,blocks(a),blocks(b))
  return a
end
