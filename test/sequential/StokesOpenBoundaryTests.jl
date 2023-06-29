module StokesOpenBoundaryTestsSeq
using PartitionedArrays
include("../StokesOpenBoundaryTests.jl")
with_debug(StokesOpenBoundaryTests.main,(2,2))
end # module
