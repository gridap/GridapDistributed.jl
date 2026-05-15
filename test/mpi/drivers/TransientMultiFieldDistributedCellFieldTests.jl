module TransientMultiFieldDistributedCellFieldMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "TransientMultiFieldDistributedCellFieldTests.jl"))

nprocs = MPI.Comm_size(MPI.COMM_WORLD)
parts = nprocs == 4 ? (2, 2) : (1, 1)

with_mpi() do distribute
  TransientMultiFieldDistributedCellFieldTests.main(distribute, parts)
end

end # module
