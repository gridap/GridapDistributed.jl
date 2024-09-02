
function permuted_variable_partition(
  n_own::AbstractArray{<:Integer},
  gids::AbstractArray{<:AbstractArray{<:Integer}},
  owners::AbstractArray{<:AbstractArray{<:Integer}};
  n_global=reduction(+,n_own,destination=:all,init=zero(eltype(n_own))),
  start=scan(+,n_own,type=:exclusive,init=one(eltype(n_own)))
)
  ranks = linear_indices(n_own)
  np = length(ranks)
  map(ranks,n_own,n_global,start,gids,owners) do rank,n_own,n_global,start,gids,owners
    n_local = length(gids)
    n_ghost = n_local - n_own
    perm = fill(zero(Int32),n_local)
    ghost_gids = fill(zero(Int),n_ghost)
    ghost_owners = fill(zero(Int32),n_ghost)

    n_ghost = 0
    for (lid,(gid,owner)) in enumerate(zip(gids,owners))
      if owner == rank
        perm[lid] = gid-start+1
      else
        n_ghost += 1
        ghost_gids[n_ghost] = gid
        ghost_owners[n_ghost] = owner
        perm[lid] = n_own + n_ghost
      end
    end
    @assert n_ghost == n_local - n_own

    ghost = GhostIndices(n_global,ghost_gids,ghost_owners)
    dof_ids = PartitionedArrays.LocalIndicesWithVariableBlockSize(
      CartesianIndex((rank,)),(np,),(n_global,),((1:n_own).+(start-1),),ghost
    )
    permute_indices(dof_ids,perm)
  end
end

function generate_ptrs(vv::AbstractArray{<:AbstractArray{T}}) where T
  ptrs = Vector{Int32}(undef,length(vv)+1)
  Arrays._generate_data_and_ptrs_fill_ptrs!(ptrs,vv)
  Arrays.length_to_ptrs!(ptrs)
  return ptrs
end
