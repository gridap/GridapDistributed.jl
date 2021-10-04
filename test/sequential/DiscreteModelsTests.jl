module DistributedDiscreteModelsTestsSeq

using PartitionedArrays

include("../DiscreteModelsTests.jl")

parts = get_part_ids(sequential,(2,2))
DiscreteModelsTests.main(parts)

parts = get_part_ids(sequential,(2,2,2))
DiscreteModelsTests.main(parts)

end
