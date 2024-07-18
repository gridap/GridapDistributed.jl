module AdaptivityTestsSeq

using PartitionedArrays

include("../CartesianAdaptivityTests.jl")
with_debug() do distribute
  CartesianAdaptivityTests.main(distribute)
end 

end # module