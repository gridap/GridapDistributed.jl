module MultiFieldTestsSeq
using PartitionedArrays
include("../MultiFieldTests.jl")
with_debug() do distribute
    MultiFieldTests.main(distribute,(2,2))
end
end # module
