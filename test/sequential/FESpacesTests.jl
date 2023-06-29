module FESpacesTestsSeq
using PartitionedArrays
include("../FESpacesTests.jl")
with_debug(FESpacesTests.main,(2,2))
end # module
