module DistributedDataTests

using GridapDistributed
using Test

nparts = 4
comm = SequentialCommunicator(nparts)

@test num_parts(comm) == nparts
@test num_workers(comm) == 1
@test comm == comm
@test i_am_master(comm)

a = DistributedData{Int}(comm) do part
  10*part
end

b = DistributedData(a) do part, a
  20*part + a
end

@test a.parts == 10*collect(1:nparts)
@test b.parts == 30*collect(1:nparts)

do_on_parts(a,b) do part, a, b
  @test a == 10*part
  @test b == 30*part
end

@test gather(b) == 30*collect(1:nparts)

c = scatter_value(comm,2)
@test c.parts == fill(2,nparts)

end # module
