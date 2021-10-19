module MultiFieldTestsSeq

using PartitionedArrays
using MPI
MPI.Init()

include("../MultiFieldTests.jl")

parts = get_part_ids(mpi,(2,2))
display(parts)
MultiFieldTests.main(parts)

end # module

