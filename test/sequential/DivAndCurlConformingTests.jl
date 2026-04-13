module DivConformingTestsSeq
using PartitionedArrays
include("../DivAndCurlConformingTests.jl")

with_debug() do distribute
  DivAndCurlConformingTests.main(distribute,2)
end

end # module
