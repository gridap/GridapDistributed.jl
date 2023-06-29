module MultiFieldTestsSeq
using PartitionedArrays
include("../MultiFieldTests.jl")
with_debug(MultiFieldTests.main,(2,2))
end # module
