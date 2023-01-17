module CellDataTestsSeq
using PartitionedArrays
include("../CellDataTests.jl")
with_backend(CellDataTests.main,SequentialBackend(),(2,2))
end # module
