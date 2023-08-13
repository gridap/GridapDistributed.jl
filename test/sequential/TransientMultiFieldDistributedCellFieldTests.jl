module TransientMultiFieldDistributedCellFieldTestsSeq
using PartitionedArrays
include("../TransientMultiFieldDistributedCellFieldTests.jl")
with_debug() do distribute
    TransientMultiFieldDistributedCellFieldTests.main(distribute,(2,2))
end 
end # module
