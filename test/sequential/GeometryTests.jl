module GeometryTestsSeq

using PartitionedArrays
using TestApp

parts = get_part_ids(sequential,(2,2))
TestApp.GeometryTests.main(parts)

parts = get_part_ids(sequential,(2,2,2))
TestApp.GeometryTests.main(parts)

end
