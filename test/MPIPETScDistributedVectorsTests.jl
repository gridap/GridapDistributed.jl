module MPIPETScDistributedVectors
using Gridap
using GridapDistributed
using Test
using MPI
using PETSc

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
  vec = DistributedVector(indices,indices) do part, indices
    local_part=Vector{Int}(undef,length(indices.lid_to_owner))
    for i=1:length(local_part)
      if (indices.lid_to_owner[i] == part)
        local_part[i]=part
      end
    end
    local_part
  end
  exchange!(vec)
  @test vec.part == indices.parts.part.lid_to_owner

  function init_vec()
    DistributedVector(indices,indices) do part, indices
      n = length(indices.lid_to_owner)
      ptrs = Vector{Int}(undef,n+1)
      ptrs[1]=1
      for i=1:n
        ptrs[i+1]=ptrs[i]+rand(1:4)
      end
      current=1
      data = Vector{Int}(undef, ptrs[n+1]-1)
      for i=1:n
        if (indices.lid_to_owner[i] == part)
          for j=1:(ptrs[i+1]-ptrs[i])
            data[current]=part
            current=current+1
          end
        else
          for j=1:(ptrs[i+1]-ptrs[i])
            data[current]=0
            current=current+1
          end
        end
      end
      Gridap.Arrays.Table(data,ptrs)
    end
  end
  vec = init_vec()
  exchange!(vec)
  function test_result()
     result = true
     for i = 1:length(vec.part)
       for j = 1:length(vec.part[i])
         result =
           result && (vec.part[i][j] == indices.parts.part.lid_to_owner[i])
       end
     end
     @test result
  end
  test_result()
end

end
