module AdaptivityTestsSeq

using PartitionedArrays

include("../AdaptivityCartesianTests.jl")
include("../AdaptivityUnstructuredTests.jl")
with_debug() do distribute
  AdaptivityCartesianTests.main(distribute)
  AdaptivityUnstructuredTests.main(distribute)
end

end # module