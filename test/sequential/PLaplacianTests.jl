module PLaplacianTestsSeq
using PartitionedArrays
include("../PLaplacianTests.jl")
prun(PLaplacianTests.main,sequential,(2,2))
end # module
