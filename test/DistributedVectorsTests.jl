module DistributedVectorsTests

using GridapDistributed
using Test

nparts = 2
comm = SequentialCommunicator(nparts)

n = 10

indices = DistributedIndexSet(comm,n) do part
  lid_to_owner = fill(part,6)
  if part == 1
    lid_to_gid = 1:6
    lid_to_owner[end] = 2
  else
    lid_to_gid = 5:10
    lid_to_owner[1] = 1
  end
  IndexSet(n,lid_to_gid,lid_to_owner)
end

a = DistributedVector(indices) do part
  fill(10*part,6)
end

exchange!(a)

@test a.parts == [[10, 10, 10, 10, 10, 20], [10, 20, 20, 20, 20, 20]]

b = DistributedVector{Int}(indices,a) do part, a
  a .+ part
end

c = b[indices]

@test c.parts == [[11, 11, 11, 11, 11, 22], [11, 22, 22, 22, 22, 22]]

v = rand(n)

w = v[indices]

do_on_parts(w,indices) do part, w, indices
  @test w == v[indices.lid_to_gid]
end

end # module
