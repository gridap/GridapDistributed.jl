module PoissonTestsSeq
using PartitionedArrays
include("../PoissonTests.jl")
with_debug(PoissonTests.main,(2,2))
end # module
