module TransientMultiFieldDistributedCellFieldTestsSeq
using PartitionedArrays
include("../../GridapODEs/TransientMultiFieldDistributedCellFieldTests.jl")
prun(TransientMultiFieldDistributedCellFieldTests.main,sequential,(2,2))
end # module
