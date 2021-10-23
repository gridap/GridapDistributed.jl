module PLaplacianTestsSeq

using PartitionedArrays

include("../PLaplacianTests.jl")

parts = get_part_ids(sequential,(2,2))
PLaplacianTests.main(parts)

end # module
