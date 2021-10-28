module FESpacesTestsSeq
using PartitionedArrays
include("../FESpacesTests.jl")
prun(FESpacesTests.main,sequential,(2,2))
end # module
