module PoissonTestsSeq
using PartitionedArrays
include("../PoissonTests.jl")
with_debug() do distribute
    PoissonTests.main(distribute,(2,2))
end 
end # module
