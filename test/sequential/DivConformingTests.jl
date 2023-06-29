module DivConformingTestsSeq
using PartitionedArrays
include("../DivConformingTests.jl")
with_debug(DivConformingTests.main,2)
end # module
