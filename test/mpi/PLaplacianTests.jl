module PLaplacianTestsSeq

using PartitionedArrays
using MPI
MPI.Init()

include("../PLaplacianTests.jl")

parts = get_part_ids(mpi,(2,2))
display(parts)
PLaplacianTests.main(parts)

end # module
