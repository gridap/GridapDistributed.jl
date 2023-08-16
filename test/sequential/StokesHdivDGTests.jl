module StokesHdivDGTestsSeq
using PartitionedArrays
include("../StokesHdivDGTests.jl")
with_debug() do distribute
    StokesHdivDGTests.main(distribute,(2,2))
end 
end # module
