module GeometryTestsSeq

using PartitionedArrays

include("../GeometryTests.jl")

parts = get_part_ids(sequential,(2,2))
GeometryTests.main(parts)

parts = get_part_ids(sequential,(2,2,2))
GeometryTests.main(parts)

end
