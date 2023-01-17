module MultiFieldTestsSeq
using PartitionedArrays
include("../MultiFieldTests.jl")
with_backend(MultiFieldTests.main,SequentialBackend(),(2,2))
end # module
