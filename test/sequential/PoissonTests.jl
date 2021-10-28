module PoissonTestsSeq
using PartitionedArrays
include("../PoissonTests.jl")
prun(PoissonTests.main,sequential,(2,2))
end # module
