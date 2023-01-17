module StokesOpenBoundaryTestsSeq
using PartitionedArrays
include("../StokesOpenBoundaryTests.jl")
with_backend(StokesOpenBoundaryTests.main,SequentialBackend(),(2,2))
end # module
