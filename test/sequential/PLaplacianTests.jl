module PLaplacianTestsSeq
using PartitionedArrays
include("../PLaplacianTests.jl")
with_debug() do distribute
    PLaplacianTests.main(distribute,(2,2))
end
end # module
