module CellDataTestsSeq

using PartitionedArrays
using TestApp

parts = get_part_ids(sequential,(2,2))
TestApp.CellDataTests.main(parts)

end # module
