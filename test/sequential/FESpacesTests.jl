module FESpacesTestsSeq
using PartitionedArrays
include("../FESpacesTests.jl")
with_debug() do distribute
  FESpacesTests.main(distribute,(2,2))
end
end # module
