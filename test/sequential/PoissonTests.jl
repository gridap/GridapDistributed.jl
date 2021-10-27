module PoissonTestsSeq

using PartitionedArrays

include("../PoissonTests.jl")

parts = get_part_ids(sequential,(2,2))
PoissonTests.main(parts)

end # module
