module TransientDistributedCellFieldTestsSeq
using PartitionedArrays
include("../TransientDistributedCellFieldTests.jl")
with_backend(TransientDistributedCellFieldTests.main,SequentialBackend(),(2,2))
end # module
