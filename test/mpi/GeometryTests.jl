module GeometryTestsSeq
using MPI
using PartitionedArrays

include("../GeometryTests.jl")

MPI.Init()

if MPI.Comm_size(MPI.COMM_WORLD) == 4
  parts = get_part_ids(mpi,(2,2))
elseif MPI.Comm_size(MPI.COMM_WORLD) == 8
  parts = get_part_ids(mpi,(2,2,2))
else
  error()
end
display(parts)
GeometryTests.main(parts)

end
