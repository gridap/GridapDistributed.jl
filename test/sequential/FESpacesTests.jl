module FESpacesTestsSeq

using PartitionedArrays
using TestApp

parts = get_part_ids(sequential,(2,2))
TestApp.FESpacesTests.main(parts)

end # module
