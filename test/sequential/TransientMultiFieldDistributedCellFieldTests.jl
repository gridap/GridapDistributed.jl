module TransientMultiFieldDistributedCellFieldTestsSeq
using PartitionedArrays
include("../TransientMultiFieldDistributedCellFieldTests.jl")
with_debug(TransientMultiFieldDistributedCellFieldTests.main,(2,2))
end # module
