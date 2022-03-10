module StokesOpenBoundaryTestsSeq
using PartitionedArrays
include("../../ODEs/StokesOpenBoundaryTests.jl")
prun(StokesOpenBoundaryTests.main,sequential,(2,2))
end # module
