module PoissonTestsSeq

using PartitionedArrays
using MPI
MPI.Init()

include("../PoissonTests.jl")

parts = get_part_ids(mpi,(2,2))
display(parts)
PoissonTests.main(parts)

end # module


