module DivConformingTestsSeq
using PartitionedArrays
include("../DivConformingTests.jl")
prun(DivConformingTests.main,sequential,2)
end # module
