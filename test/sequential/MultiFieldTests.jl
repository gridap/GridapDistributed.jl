module MultiFieldTestsSeq

using PartitionedArrays

include("../MultiFieldTests.jl")

parts = get_part_ids(sequential,(2,2))
MultiFieldTests.main(parts)

end # module
