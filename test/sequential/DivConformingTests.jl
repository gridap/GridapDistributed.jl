module DivConformingTestsSeq
using PartitionedArrays
include("../DivConformingTests.jl")
with_backend(DivConformingTests.main,SequentialBackend(),2)
end # module
