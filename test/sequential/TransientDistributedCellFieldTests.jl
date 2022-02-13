module TransientDistributedCellFieldTestsSeq
using PartitionedArrays
include("../TransientDistributedCellFieldTests.jl")
prun(TransientDistributedCellFieldTests.main,sequential,(2,2))
end # module
