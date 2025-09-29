module ZeroMeanFESpacesTestsSeq

using PartitionedArrays
include("../ZeroMeanFESpacesTests.jl")

with_debug() do distribute
  ZeroMeanFESpacesTests.main(distribute,(2,2))
end

end # module
