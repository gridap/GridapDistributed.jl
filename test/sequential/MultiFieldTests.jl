module MultiFieldTestsSeq

using PartitionedArrays
using TestApp

parts = get_part_ids(sequential,(2,2))
TestApp.MultiFieldTests.main(parts)

end # module
