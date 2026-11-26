module HcurlProjectionTestsSeq
using PartitionedArrays

include("../HcurlProjectionTests.jl")

# The 3D branch of these tests is the one that exercises the face DoFs of
# `nedelec` on hexahedra, and therefore the sign conventions of
# `compute_cell_bases_changes`. It used to be unreachable from CI, since the MPI
# drivers only ever passed a 2-tuple of parts.
with_debug() do distribute
  HcurlProjectionTests.main(distribute,(1,1,2))
end

end # module
