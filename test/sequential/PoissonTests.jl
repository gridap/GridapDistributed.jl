module PoissonTestsSeq

using TestApp
using PartitionedArrays

parts = get_part_ids(sequential,(2,2))
TestApp.PoissonTests.main(parts)

end # module
