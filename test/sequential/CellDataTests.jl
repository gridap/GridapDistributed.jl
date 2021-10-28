module CellDataTestsSeq
using PartitionedArrays
include("../CellDataTests.jl")
prun(CellDataTests.main,sequential,(2,2))
end # module
