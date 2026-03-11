module BlockPartitionedArraysTests

using Test, LinearAlgebra, BlockArrays, SparseArrays

using Gridap
using GridapDistributed
using PartitionedArrays
using GridapDistributed: BlockPArray, BlockPVector, BlockPMatrix, BlockPRange

function main(distribute, parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  n_global = 4*prod(parts)
  indices = uniform_partition(ranks, n_global, true)

  pr = PRange(indices)
  block_range = BlockPRange([pr, pr])

  # BlockPRange API
  @test blocklength(block_range) == 2
  @test blocksize(block_range) == (2,)

  # BlockPVector: constructor, fill!, similar, arithmetic
  v = BlockPVector{Vector{Float64}}(undef, block_range)
  fill!(v, 1.0)
  v2 = similar(v, block_range)
  fill!(v2, 2.0)

  v2 = v .+ 1.0
  v2 = v .- 1.0
  v2 = v .* 2.0
  v2 = v ./ 2.0
  copy!(v2, v)

  @test norm(v) ≈ sqrt(2*n_global)

  # PartitionedArrays vector API
  consistent!(v) |> wait
  assemble!(v)   |> wait
  partition(v)
  local_values(v)
  own_values(v)
  ghost_values(v)

  # local_views on vector with and without new range
  local_views(v)
  local_views(v, block_range)

  # BlockPMatrix: constructor (exercises BUG3 fix: cols must not be dropped)
  m = BlockPMatrix{SparseMatrixCSC{Float64,Int64}}(undef, block_range, block_range)
  @test blocksize(m) == (2,2)

  m2 = similar(m, (block_range, block_range))
  @test blocksize(m2) == (2,2)

  # PartitionedArrays matrix API
  assemble!(m)  |> wait
  LinearAlgebra.fillstored!(m, 0.0)
  partition(m)
  own_ghost_values(m)
  ghost_own_values(m)

  # local_views on matrix (exercises BUG5 fix: correct I[1]/I[2] indexing)
  local_views(m)
  local_views(m, block_range, block_range)

  maximum(abs, v)
  minimum(abs, v)

  # BUG4 fix: mul!(y,A,x,α,β) must compute α*(A*x) + β*y, not α*β*(A*x)
  # With A=0: y ← α*(0*x) + β*y_old = β*y_old
  fill!(v, 1.0)
  v4 = similar(v, block_range)
  fill!(v4, 3.0)
  LinearAlgebra.fillstored!(m, 0.0)
  mul!(v4, m, v, 2.0, 2.0)   # y ← 2*(0*x) + 2*[3,...] = [6,...]
  @test norm(v4) ≈ 6.0 * sqrt(2*n_global)
end

end # module
