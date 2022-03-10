module TransientDistributedCellFieldTestsSeq
using PartitionedArrays
include("../../ODEs/TransientDistributedCellFieldTests.jl")
prun(TransientDistributedCellFieldTests.main,sequential,(2,2))
end # module
