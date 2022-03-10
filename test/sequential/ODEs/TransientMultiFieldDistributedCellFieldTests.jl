module TransientMultiFieldDistributedCellFieldTestsSeq
using PartitionedArrays
include("../../ODEs/TransientMultiFieldDistributedCellFieldTests.jl")
prun(TransientMultiFieldDistributedCellFieldTests.main,sequential,(2,2))
end # module
