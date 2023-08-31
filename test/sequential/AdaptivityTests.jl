module AdaptivityTestsSeq
using PartitionedArrays
include("../AdaptivityTests.jl")

with_debug() do distribute
  AdaptivityTests.main(distribute)
end

end # module