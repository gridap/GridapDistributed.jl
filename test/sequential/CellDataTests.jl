module CellDataTestsSeq
using PartitionedArrays
include("../CellDataTests.jl")
with_debug() do distribute
    CellDataTests.main(distribute,(2,2))
end
end # module
