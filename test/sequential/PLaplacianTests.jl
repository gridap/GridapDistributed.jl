module PLaplacianTestsSeq
using PartitionedArrays
include("../PLaplacianTests.jl")
with_debug(PLaplacianTests.main,(2,2))
end # module
