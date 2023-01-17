module PLaplacianTestsSeq
using PartitionedArrays
include("../PLaplacianTests.jl")
with_backend(PLaplacianTests.main,SequentialBackend(),(2,2))
end # module
