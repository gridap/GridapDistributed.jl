module NP4
# All test running on 4 procs here

using TestApp
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

if ! MPI.Initialized()
  MPI.Init()
end

include("runtests_np4_body.jl")

if MPI.Comm_size(MPI.COMM_WORLD) == 4
  with_backend(all_tests,MPIBackend(),(2,2))
elseif MPI.Comm_size(MPI.COMM_WORLD) == 1
  with_backend(all_tests,MPIBackend(),(1,1))
else
  MPI.Abort(MPI.COMM_WORLD,0)
end

end #module
