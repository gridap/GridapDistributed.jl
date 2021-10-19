module FESpacesTestsSeq

using PartitionedArrays
using MPI
MPI.Init()

include("../FESpacesTests.jl")

parts = get_part_ids(mpi,(2,2))
display(parts)
FESpacesTests.main(parts)

end # module

