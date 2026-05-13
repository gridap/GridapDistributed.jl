module BlockSparseMatrixAssemblersTestsSeq
using PartitionedArrays
include("../BlockSparseMatrixAssemblersTests.jl")
include("../BlockPartitionedArraysTests.jl")

with_debug() do distribute
  BlockSparseMatrixAssemblersTests.main(distribute,(2,2))
  BlockPartitionedArraysTests.main(distribute,(2,2))
end

with_debug() do distribute
  BlockSparseMatrixAssemblersTests.main(distribute,(2,1))
  BlockPartitionedArraysTests.main(distribute,(2,1))
end

with_debug() do distribute
  BlockSparseMatrixAssemblersTests.main(distribute,(1,2))
  BlockPartitionedArraysTests.main(distribute,(1,2))
end

end # module