module StokesOpenBoundaryMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "StokesOpenBoundaryTests.jl"))

with_mpi() do distribute
  nprocs = MPI.Comm_size(MPI.COMM_WORLD)
  parts = nprocs == 4 ? (2, 2) : (1, 1)
  StokesOpenBoundaryTests.main(distribute, parts)
end

end # module
