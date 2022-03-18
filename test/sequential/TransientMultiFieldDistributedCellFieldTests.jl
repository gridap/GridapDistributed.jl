module TransientMultiFieldDistributedCellFieldTestsSeq
using PartitionedArrays
include("../TransientMultiFieldDistributedCellFieldTests.jl")
prun(TransientMultiFieldDistributedCellFieldTests.main,sequential,(2,2))
end # module
