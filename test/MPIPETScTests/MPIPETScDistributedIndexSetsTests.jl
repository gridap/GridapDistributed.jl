module MPIPETScDistributedIndexSets

using GridapDistributed
using Test
using MPI

MPIPETScCommunicator() do comm
  @test num_parts(comm) == 2

  n = 10
  indices = DistributedIndexSet(comm,n) do part
    if part == 1
      IndexSet(n,1:5,[2,1,1,1,2])
    else
      IndexSet(n,[2,6,5,7,8,9,10,1],[1,2,2,2,2,2,2,2])
    end
  end

  #do_on_parts(indices) do part, indices
  #  @test indices.lid_to_owner == fill(part,5)
  #  @test indices.ngids == n
  #end

  println("$(MPI.Comm_rank(comm.comm)): lid_to_gid_petsc=$(indices.lid_to_gid_petsc)
                                      petsc_to_app_locidx=$(indices.lid_to_gid_petsc)
                                      app_to_petsc_locidx=$(indices.lid_to_gid_petsc)")
  #@test get_comm(indices) === comm
  #@test num_parts(indices) == nparts
  #@test num_gids(indices) == n
end

end # module
