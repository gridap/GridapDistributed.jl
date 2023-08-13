module StokesOpenBoundaryTestsSeq
using PartitionedArrays
include("../StokesOpenBoundaryTests.jl")
with_debug() do distribute
  StokesOpenBoundaryTests.main(distribute,(2,2))
end
end # module
