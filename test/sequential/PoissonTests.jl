module PoissonTestsSeq
using PartitionedArrays
include("../PoissonTests.jl")
with_backend(PoissonTests.main,SequentialBackend(),(2,2))
end # module
