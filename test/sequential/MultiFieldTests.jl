module MultiFieldTestsSeq
using PartitionedArrays
include("../MultiFieldTests.jl")
prun(MultiFieldTests.main,sequential,(2,2))
end # module
