module DivConformingTestsSeq
using PartitionedArrays
include("../DivConformingTests.jl")

with_debug() do distribute
  DivConformingTests.main(distribute,2)
end

end # module
