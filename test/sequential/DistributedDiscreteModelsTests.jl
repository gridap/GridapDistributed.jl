module DistributedDiscreteModelsTestsSeq

using PartitionedArrays

include("../DistributedDiscreteModelsTests.jl")

parts = get_part_ids(sequential,(2,2))
DistributedDiscreteModelsTests.main(parts)

parts = get_part_ids(sequential,(2,2,2))
DistributedDiscreteModelsTests.main(parts)

end
