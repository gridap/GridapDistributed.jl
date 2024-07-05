module AdaptivityTestsSeq

using PartitionedArrays

include("../AdaptivityCartesianTests.jl")
with_debug() do distribute
  AdaptivityCartesianTests.main(distribute)
end 

end # module