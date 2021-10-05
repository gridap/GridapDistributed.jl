module TriangulationsTestsSeq

using PartitionedArrays

include("../TriangulationsTests.jl")

parts = get_part_ids(sequential,(2,2))
TriangulationsTests.main(parts)

#parts = get_part_ids(sequential,(2,2,2))
#TriangulationsTests.main(parts)

end

