module FESpacesTestsSeq
using PartitionedArrays
include("../FESpacesTests.jl")
with_backend(FESpacesTests.main,SequentialBackend(),(2,2))
end # module
