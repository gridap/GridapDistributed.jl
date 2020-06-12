module MPIPETScDistributedVectors

using GridapDistributed
using Test
using MPI

comm = MPIPETScCommunicator()
@test num_parts(comm) == 2

n = 10
indices = DistributedIndexSet(comm,n) do part
  if part == 1
    IndexSet(n,1:5,[2,1,1,1,2])
  else
    IndexSet(n,[2,6,5,7,8,9,10,1],[1,2,2,2,2,2,2,2])
  end
end

vec = DistributedVector(indices,indices) do part, indices
  local_part=Vector{Int64}(undef,length(indices.lid_to_owner))
  for i=1:length(local_part)
    if (indices.lid_to_owner[i] == part)
      local_part[i]=part
    end
  end
  local_part
end
exchange!(vec)
@test vec.part == indices.parts.part.lid_to_owner

vec = DistributedVector(indices,indices) do part, indices
  local_part=Vector{Vector{Int64}}(undef,length(indices.lid_to_owner))
  for i=1:length(local_part)
    if (indices.lid_to_owner[i] == part)
      local_part[i]=[part for j=1:4]
    else
      local_part[i]=[0 for j=1:4]
    end
  end
  local_part
end
exchange!(vec)

test_result = true
for i = 1:length(vec.part)
  for j = 1:length(vec.part[i])
     global test_result
     test_result = test_result && (vec.part[i][j] == indices.parts.part.lid_to_owner[i])
  end
end

@test test_result

end # module