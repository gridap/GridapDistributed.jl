
using Test
using Gridap
using PartitionedArrays
using GridapDistributed
using BlockArrays
using SparseArrays
using LinearAlgebra

using GridapDistributed: BlockPArray, BlockPVector, BlockPMatrix, BlockPRange


ranks = with_debug() do distribute
  distribute(LinearIndices((2,)))
end

indices = map(ranks) do r
  if r == 1
    own_gids = [1,2,3,4,5]
    ghost_gids   = [6,7]
    ghost_owners = [2,2]
  else
    own_gids = [6,7,8,9,10]
    ghost_gids   = [5]
    ghost_owners = [1]
  end
  own_idx   = OwnIndices(10,r,own_gids)
  ghost_idx = GhostIndices(10,ghost_gids,ghost_owners)
  OwnAndGhostIndices(own_idx,ghost_idx)
end

block_range = BlockPRange([PRange(indices),PRange(indices)])

_v = PVector{OwnAndGhostVectors{Vector{Float64}}}(undef,indices)
v = BlockPArray([_v,_v],(block_range,))

_m = map(CartesianIndices((2,2))) do I
  i,j = I[1],I[2]
  local_mats = map(ranks,indices) do r, ind
    n = local_length(ind)
    if i==j && r == 1
      SparseMatrixCSC(n,n,Int[1,3,5,7,9,10,11,13],Int[1,2,2,3,3,4,4,5,5,6,6,7],fill(1.0,12))
    elseif i==j && r == 2
      SparseMatrixCSC(n,n,Int[1,2,4,6,8,10,11],Int[1,1,2,2,3,3,4,4,5,6],fill(1.0,10))
    else
      SparseMatrixCSC(n,n,fill(Int(1),n+1),Int[],Float64.([]))
    end
  end
  PSparseMatrix(local_mats,indices,indices)
end
m = BlockPArray(_m,(block_range,block_range))

x = similar(_v)
mul!(x,_m[1,1],_v)

# BlockPRange

@test blocklength(block_range) == 2
@test blocksize(block_range) == (2,)

# BlockPArray

__v = similar(v,block_range)
__m = similar(m,(block_range,block_range))
fill!(v,1.0)

__v = __v .+ 1.0
__v = __v .- 1.0
__v = __v .* 1.0
__v = __v ./ 1.0

__m = __m .+ 1.0
__m = __m .- 1.0
__m = __m .* 1.0
__m = __m ./ 1.0

# LinearAlgebra

x = similar(v)
mul!(x,m,v)
consistent!(x) |> fetch
partition(x)

dot(v,x)
norm(v)
copy!(x,v)

LinearAlgebra.fillstored!(__m,1.0)

__v = BlockPVector{Float64,PVector{Vector{Float64}}}(undef,block_range)

m[Block(1,1)]
m[Block(1),Block(1)]

