module TransientMultiFieldDistributedCellFieldTestsSeq
using PartitionedArrays
include("../TransientMultiFieldDistributedCellFieldTests.jl")
with_backend(TransientMultiFieldDistributedCellFieldTests.main,SequentialBackend(),(2,2))
end # module
