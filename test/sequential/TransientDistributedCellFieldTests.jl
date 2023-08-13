module TransientDistributedCellFieldTestsSeq
using PartitionedArrays
include("../TransientDistributedCellFieldTests.jl")
with_debug() do distribute
    TransientDistributedCellFieldTests.main(distribute,(2,2))
end
end # module
