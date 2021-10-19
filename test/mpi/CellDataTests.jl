module CellDataTestsSeq

using PartitionedArrays
using MPI
MPI.Init()

include("../CellDataTests.jl")

parts = get_part_ids(mpi,(2,2))
display(parts)
CellDataTests.main(parts)

end # module
