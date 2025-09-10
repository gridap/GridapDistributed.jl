
# This is copied from PArrays v0.5, which is not yet supported. It is necessary to get 
# more than one layers of ghosts.

function _uniform_partition(rank,np,n,args...)
  @assert prod(np) == length(rank)
  indices = map(rank) do rank
    _block_with_constant_size(rank,np,n,args...)
  end
  if length(args) == 0
    map(indices) do indices
      cache = assembly_cache(indices)
      copy!(cache,empty_assembly_cache())
    end
  else
    assembly_neighbors(indices;symmetric=true)
  end
  indices
end

function _block_with_constant_size(rank,np,n,ghost,periodic=map(i->false,ghost))
  N = length(n)
  p = CartesianIndices(np)[rank]
  own_ranges = map(_local_range,Tuple(p),np,n)
  local_ranges = map(_local_range,Tuple(p),np,n,ghost,periodic)
  owners = map(Tuple(p), np, n, local_ranges) do p, np, n, lr
    myowners = zeros(Int32,length(lr))
    i = 1
    for p in Iterators.cycle(1:np)
      plr = _local_range(p, np, n)
      while mod(lr[i]-1, n)+1 in plr
        myowners[i] = p
        (i += 1) > length(myowners) && return myowners
      end
    end
  end
  n_local = prod(map(length, local_ranges))
  n_own = prod(map(length, own_ranges))
  n_ghost = n_local - n_own

  ghost_to_global = zeros(Int,n_ghost)
  ghost_to_owner = zeros(Int32,n_ghost)
  perm = zeros(Int32,n_local)
  i_ghost = 0
  i_own = 0

  cis = CartesianIndices(map(length,local_ranges))
  lis = CircularArray(LinearIndices(n))
  local_cis = CartesianIndices(local_ranges)
  owner_lis = LinearIndices(np)
  for (i,ci) in enumerate(cis)
    flags = map(Tuple(ci), own_ranges, local_ranges) do i, or, lr
      i in (or .- first(lr) .+ 1)
    end
    if !all(flags)
      i_ghost += 1
      ghost_to_global[i_ghost] = lis[local_cis[i]]
      o = map(getindex,owners,Tuple(ci))
      o_ci = CartesianIndex(o)
      ghost_to_owner[i_ghost] = owner_lis[o_ci]
      perm[i] = i_ghost + n_own
    else
      i_own += 1
      perm[i] = i_own
    end
  end
  ghostids = GhostIndices(prod(n),ghost_to_global,ghost_to_owner)
  ids = PartitionedArrays.LocalIndicesWithConstantBlockSize(p,np,n,ghostids)
  PartitionedArrays.PermutedLocalIndices(ids,perm)
end

function _local_range(p,np,n,ghost=false,periodic=false)
  l, rem = divrem(n, np)
  offset = l * (p-1)
  if rem >= (np-p+1)
    l += 1
    offset += p - (np-rem) - 1
  end
  start = 1+offset-ghost
  stop = l+offset+ghost

  periodic && return start:stop
  return max(1, start):min(n,stop)
end
