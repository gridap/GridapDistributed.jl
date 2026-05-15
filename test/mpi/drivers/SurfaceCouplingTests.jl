module SurfaceCouplingMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "SurfaceCouplingTests.jl"))

nprocs = MPI.Comm_size(MPI.COMM_WORLD)
parts = nprocs == 4 ? (2, 2) : (1, 1)

with_mpi() do distribute
  SurfaceCouplingTests.main(distribute, parts)
end

end # module
