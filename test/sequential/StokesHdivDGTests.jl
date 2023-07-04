module StokesHdivDGTests
using PartitionedArrays
include("../StokesHdivDGTests.jl")
with_debug() do distribute
    StokesHdivDGTests.main(distribute,(1,1))
end 
end # module
