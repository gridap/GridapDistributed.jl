module DistributedIndexSets

using GridapDistributed
using Test

nparts = 2
comm = SequentialCommunicator(nparts)

n = 10

indices = DistributedIndexSet(comm,n) do part
  if part == 1
    IndexSet(n,1:5,fill(part,5))
  else
    IndexSet(n,6:10,fill(part,5))
  end
end

do_on_parts(indices) do part, indices
  @test indices.lid_to_owner == fill(part,5)
  @test indices.ngids == n
end

@test get_comm(indices) === comm
@test num_parts(indices) == nparts
@test num_gids(indices) == n

end # module
