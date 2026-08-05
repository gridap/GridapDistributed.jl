module TransientDistributedCellFieldMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "TransientDistributedCellFieldTests.jl"))

with_mpi() do distribute
  nprocs = MPI.Comm_size(MPI.COMM_WORLD)
  parts = nprocs == 4 ? (2, 2) : (1, 1)
  TransientDistributedCellFieldTests.main(distribute, parts)
end

end # module
