module PoissonTestsSeq

using PartitionedArrays

include("../src/PoissonTests.jl")

parts = get_part_ids(sequential,(2,2))
PoissonTests.main(parts)

end # module
