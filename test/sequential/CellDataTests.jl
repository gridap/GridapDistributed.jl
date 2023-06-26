module CellDataTestsSeq
using PartitionedArrays
include("../CellDataTests.jl")
with_debug(CellDataTests.main,(2,2))
end # module
