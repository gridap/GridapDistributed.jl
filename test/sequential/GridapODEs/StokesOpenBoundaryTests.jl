module StokesOpenBoundaryTestsSeq
using PartitionedArrays
include("../../GridapODEs/StokesOpenBoundaryTests.jl")
prun(StokesOpenBoundaryTests.main,sequential,(2,2))
end # module
