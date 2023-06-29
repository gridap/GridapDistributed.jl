module TransientDistributedCellFieldTestsSeq
using PartitionedArrays
include("../TransientDistributedCellFieldTests.jl")
with_debug(TransientDistributedCellFieldTests.main,(2,2))
end # module
