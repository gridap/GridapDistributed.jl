module AdaptivityTestsSeq

using PartitionedArrays

include("../AdaptivityCartesianTests.jl")
include("../AdaptivityUnstructuredTests.jl")
include("../PolytopalCoarseningTests.jl")
with_debug() do distribute
  AdaptivityCartesianTests.main(distribute)
  AdaptivityUnstructuredTests.main(distribute)
  PolytopalCoarseningTests.main(distribute,(2,2))
end

end # module