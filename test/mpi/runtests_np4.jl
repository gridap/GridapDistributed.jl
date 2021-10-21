module NP4
# All test running on 4 procs here

using TestApp
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

if ! MPI.Initialized()
  MPI.Init()
end

if MPI.Comm_size(MPI.COMM_WORLD) == 4
  parts = get_part_ids(mpi,(2,2))
elseif MPI.Comm_size(MPI.COMM_WORLD) == 1
  parts = get_part_ids(mpi,(1,1))
else
  error()
end

include("runtests_np4_body.jl")

MPI.Finalize()

end #module
