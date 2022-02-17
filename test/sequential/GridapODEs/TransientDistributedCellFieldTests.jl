module TransientDistributedCellFieldTestsSeq
using PartitionedArrays
include("../../GridapODEs/TransientDistributedCellFieldTests.jl")
prun(TransientDistributedCellFieldTests.main,sequential,(2,2))
end # module
