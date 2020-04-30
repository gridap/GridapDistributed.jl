module DistributedIndexSets

using GridapDistributed
using Test

nparts = 2
comm = SequentialCommunicator(nparts)

n = 10

indices = DistributedIndexSet(comm) do part
  if part == 1
    IndexSet(1:5,fill(part,5))
  else
    IndexSet(6:10,fill(part,5))
  end
end

do_on_parts(indices) do part, indices
  @test indices.lid_to_owner == fill(part,5)
end

@test get_comm(indices) === comm
@test num_parts(indices) == nparts

end # module
