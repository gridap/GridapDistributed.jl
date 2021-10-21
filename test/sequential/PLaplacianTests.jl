module PLaplacianTestsSeq

using PartitionedArrays
using TestApp

parts = get_part_ids(sequential,(2,2))
TestApp.LaplacianTests.main(parts)

end # module
