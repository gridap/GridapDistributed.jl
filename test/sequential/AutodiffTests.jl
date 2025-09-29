module AutodiffTestsSeq
using PartitionedArrays
include("../AutodiffTests.jl")
with_debug() do distribute
  AutodiffTests.main(distribute,(2,2))
end
end # module
