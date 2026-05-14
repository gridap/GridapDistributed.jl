
# Cell-wise communication helpers

function dof_wise_to_cell_wise!(cell_wise_vector,dof_wise_vector,cell_to_ldofs,cell_ids)
  map(cell_wise_vector,dof_wise_vector,cell_to_ldofs,cell_ids) do cwv,dwv,cell_to_ldofs,cell_ids
    cache  = array_cache(cell_to_ldofs)
    for cell in cell_ids
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      p = cwv.ptrs[cell]-1
      for (i,ldof) in enumerate(ldofs)
        if ldof > 0
          cwv.data[i+p] = dwv[ldof]
        end
      end
    end
  end
  return cell_wise_vector
end

function posneg_wise_to_cell_wise!(cell_wise_vector,pos_wise_vector,neg_wise_vector,cell_to_posneg,cell_ids)
  map(cell_wise_vector,pos_wise_vector,neg_wise_vector,cell_to_posneg,cell_ids) do cwv,pwv,nwv,cell_to_posneg,cell_ids
    cache  = array_cache(cell_to_posneg)
    for cell in cell_ids
      lids = getindex!(cache,cell_to_posneg,cell)
      p = cwv.ptrs[cell]-1
      for (i,lid) in enumerate(lids)
        if lid > 0
          cwv.data[i+p] = pwv[lid]
        elseif lid < 0
          cwv.data[i+p] = nwv[-lid]
        end
      end
    end
  end
  return cell_wise_vector
end

function cell_wise_to_dof_wise!(dof_wise_vector,cell_wise_vector,cell_to_ldofs,cell_ids)
  map(dof_wise_vector,cell_wise_vector,cell_to_ldofs,cell_ids) do dwv,cwv,cell_to_ldofs,cell_ids
    cache = array_cache(cell_to_ldofs)
    for cell in cell_ids
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      p = cwv.ptrs[cell]-1
      for (i,ldof) in enumerate(ldofs)
        if ldof > 0
          dwv[ldof] = cwv.data[i+p]
        end
      end
    end
  end
  return dof_wise_vector
end

function cell_wise_to_posneg_wise!(pos_wise_vector,neg_wise_vector,cell_wise_vector,cell_to_posneg,cell_ids)
  map(pos_wise_vector,neg_wise_vector,cell_wise_vector,cell_to_posneg,cell_ids) do pwv,nwv,cwv,cell_to_posneg,cell_ids
    cache = array_cache(cell_to_posneg)
    for cell in cell_ids
      lids = getindex!(cache,cell_to_posneg,cell)
      p = cwv.ptrs[cell]-1
      for (i,lid) in enumerate(lids)
        if lid > 0
          pwv[lid] = cwv.data[i+p]
        elseif lid < 0
          nwv[-lid] = cwv.data[i+p]
        end
      end
    end
  end
  return pos_wise_vector, neg_wise_vector
end

function allocate_cell_wise_vector(T, cell_to_lids)
  map(cell_to_lids) do cell_to_lids
    ptrs = Arrays.generate_ptrs(cell_to_lids)
    data = zeros(T,ptrs[end]-1)
    JaggedArray(data,ptrs)
  end
end

function dof_wise_to_cell_wise(
  dof_wise_vector, cell_to_ldofs, cell_ids;
  T = eltype(eltype(dof_wise_vector))
)
  cwv = allocate_cell_wise_vector(T,cell_to_ldofs)
  dof_wise_to_cell_wise!(cwv,dof_wise_vector,cell_to_ldofs,cell_ids)
  return cwv
end

function fetch_vector_ghost_values_cache(vector_partition,partition)
  cache = PArrays.p_vector_cache(vector_partition,partition)
  reverse(cache)
end

function fetch_vector_ghost_values!(vector_partition,cache)
  assemble!((a,b)->b, vector_partition, cache)
end

# PRange generation from cell meshes

"""
    generate_gids(cell_gids::PRange, cell_to_lids, nlids) -> PRange

Given a set of cell global ids, a distributed array of local tables mapping cells to local dof ids,
and a distributed array with the number of local dofs in each partition, this function
generates the global dof ids and returns them as a `PRange` of `PermutedLocalIndices`.

Ignores negative local dof ids (usually used for Dirichlet dofs).
"""
function generate_gids(
  cell_range::PRange,
  cell_to_ldofs::AbstractArray{<:AbstractArray},
  nldofs::AbstractArray{<:Integer}
)
  ranks = linear_indices(partition(cell_range))
  cell_ldofs_to_data = allocate_cell_wise_vector(Int, cell_to_ldofs)
  cache_fetch = fetch_vector_ghost_values_cache(cell_ldofs_to_data,partition(cell_range))

  # Find and count number owned dofs
  ldof_to_owner, nodofs = map(partition(cell_range),cell_to_ldofs,nldofs) do indices,cell_to_ldofs,nldofs
    ldof_to_owner = fill(Int32(0),nldofs)
    cache = array_cache(cell_to_ldofs)
    for (cell, owner) in enumerate(local_to_owner(indices))
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      for ldof in ldofs
        if ldof > 0
          # NOTE: this approach concentrates dofs in the last processor
          ldof_to_owner[ldof] = max(owner,ldof_to_owner[ldof])
        end
      end
    end
    nodofs = count(isequal(part_id(indices)),ldof_to_owner)
    ldof_to_owner, nodofs
  end |> tuple_of_arrays

  # Find the global range of owned dofs
  first_gdof = scan(+,nodofs,type=:exclusive,init=one(eltype(nodofs)))

  # Exchange the dof owners. Cell owner always has correct dof owner.
  cell_ldofs_to_owner = dof_wise_to_cell_wise!(
    cell_ldofs_to_data,ldof_to_owner,cell_to_ldofs,own_to_local(cell_range)
  )
  fetch_vector_ghost_values!(cell_ldofs_to_owner,cache_fetch) |> wait
  cell_wise_to_dof_wise!(
    ldof_to_owner,cell_ldofs_to_owner,cell_to_ldofs,ghost_to_local(cell_range)
  )

  # Fill owned gids
  ldof_to_gdof = map(ranks,first_gdof,ldof_to_owner) do rank,first_gdof,ldof_to_owner
    offset = first_gdof-1
    ldof_to_gdof = zeros(Int,length(ldof_to_owner))
    odof = 0
    for (ldof,owner) in enumerate(ldof_to_owner)
      if owner == rank
        odof += 1
        ldof_to_gdof[ldof] = odof + offset
      end
    end
    ldof_to_gdof
  end

  # Exchange gids
  cell_to_gdofs = dof_wise_to_cell_wise!(
    cell_ldofs_to_data,ldof_to_gdof,cell_to_ldofs,own_to_local(cell_range)
  )
  fetch_vector_ghost_values!(cell_to_gdofs,cache_fetch) |> wait

  # Fill ghost gids with exchanged information
  map(
    cell_to_ldofs,cell_to_gdofs,ldof_to_gdof,ldof_to_owner,partition(cell_range)
  ) do cell_to_ldofs,cell_to_gdofs,ldof_to_gdof,ldof_to_owner,indices
    cache = array_cache(cell_to_ldofs)
    lcell_to_owner = local_to_owner(indices)
    for cell in ghost_to_local(indices)
      p = cell_to_gdofs.ptrs[cell]-1
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      cell_owner = lcell_to_owner[cell]
      for (i,ldof) in enumerate(ldofs)
        if (ldof > 0) && isequal(ldof_to_owner[ldof],cell_owner)
          ldof_to_gdof[ldof] = cell_to_gdofs.data[i+p]
        end
      end
    end
  end

  dof_wise_to_cell_wise!(cell_to_gdofs,ldof_to_gdof,cell_to_ldofs,own_to_local(cell_range))
  fetch_vector_ghost_values!(cell_to_gdofs,cache_fetch) |> wait
  cell_wise_to_dof_wise!(ldof_to_gdof,cell_to_gdofs,cell_to_ldofs,ghost_to_local(cell_range))

  # Setup DoFs LocalIndices
  local_indices = permuted_variable_partition(
    nodofs,ldof_to_gdof,ldof_to_owner;start=first_gdof
  )

  return PRange(local_indices)
end

"""
    generate_posneg_gids(cell_gids::PRange, cell_to_lposneg, nlpos, nlneg) -> (PRange,PRange)

Similar to `generate_gids`, but also handles negative local dof ids. Returns two sets of
global ids: one for positive local ids and another for negative local ids.
This can be used to generate simultaneously free and dirichlet dof global ids.
"""
function generate_posneg_gids(
  cell_range::PRange,
  cell_to_lposneg::AbstractArray{<:AbstractArray},
  nlpos::AbstractArray{<:Integer},
  nlneg::AbstractArray{<:Integer}
)
  ranks = linear_indices(partition(cell_range))
  cell_lids_to_data = allocate_cell_wise_vector(Int, cell_to_lposneg)
  cache_fetch = fetch_vector_ghost_values_cache(cell_lids_to_data,partition(cell_range))

  # Find and count number owned dofs
  lpos_to_owner, lneg_to_owner, nopos, noneg = map(
    partition(cell_range),cell_to_lposneg,nlpos,nlneg
  ) do indices,cell_to_lposneg,nlpos,nlneg
    lpos_to_owner = fill(zero(Int32),nlpos)
    lneg_to_owner = fill(zero(Int32),nlneg)
    cache = array_cache(cell_to_lposneg)
    for (cell, owner) in enumerate(local_to_owner(indices))
      lids = getindex!(cache,cell_to_lposneg,cell)
      for lid in lids
        if lid > 0
          lpos_to_owner[lid] = max(owner,lpos_to_owner[lid])
        elseif lid < 0
          lneg_to_owner[-lid] = max(owner,lneg_to_owner[-lid])
        end
      end
    end
    rank = part_id(indices)
    nopos = count(isequal(rank),lpos_to_owner)
    noneg = count(isequal(rank),lneg_to_owner)
    return lpos_to_owner, lneg_to_owner, nopos, noneg
  end |> tuple_of_arrays

  # Find the global range of owned dofs
  first_gpos = scan(+,nopos,type=:exclusive,init=1)
  first_gneg = scan(+,noneg,type=:exclusive,init=1)

  # Exchange the dof owners. Cell owner always has correct dof owner.
  cell_ldofs_to_owner = posneg_wise_to_cell_wise!(
    cell_lids_to_data,lpos_to_owner,lneg_to_owner,cell_to_lposneg,own_to_local(cell_range)
  )
  fetch_vector_ghost_values!(cell_ldofs_to_owner,cache_fetch) |> wait
  cell_wise_to_posneg_wise!(
    lpos_to_owner,lneg_to_owner,cell_ldofs_to_owner,cell_to_lposneg,ghost_to_local(cell_range)
  )

  # Fill owned gids
  lpos_to_gpos = map(ranks,first_gpos,lpos_to_owner) do rank,first_gpos,lpos_to_owner
    offset = first_gpos-1
    lpos_to_gpos = zeros(Int,length(lpos_to_owner))
    opos = 0
    for (lpos,owner) in enumerate(lpos_to_owner)
      if owner == rank
        opos += 1
        lpos_to_gpos[lpos] = opos + offset
      end
    end
    lpos_to_gpos
  end

  lneg_to_gneg = map(ranks,first_gneg,lneg_to_owner) do rank,first_gneg,lneg_to_owner
    offset = first_gneg-1
    lneg_to_gneg = zeros(Int,length(lneg_to_owner))
    oneg = 0
    for (lneg,owner) in enumerate(lneg_to_owner)
      if owner == rank
        oneg += 1
        lneg_to_gneg[lneg] = oneg + offset
      end
    end
    lneg_to_gneg
  end

  # Exchange gids
  cell_to_gposneg = posneg_wise_to_cell_wise!(
    cell_lids_to_data,lpos_to_gpos,lneg_to_gneg,cell_to_lposneg,own_to_local(cell_range)
  )
  fetch_vector_ghost_values!(cell_to_gposneg,cache_fetch) |> wait

  # Fill ghost gids with exchanged information
  map(
    cell_to_lposneg,cell_to_gposneg,lpos_to_gpos,lneg_to_gneg,lpos_to_owner,lneg_to_owner,partition(cell_range)
  ) do cell_to_lposneg,cell_to_gposneg,lpos_to_gpos,lneg_to_gneg,lpos_to_owner,lneg_to_owner,indices
    cache = array_cache(cell_to_lposneg)
    lcell_to_owner = local_to_owner(indices)
    for cell in ghost_to_local(indices)
      p = cell_to_gposneg.ptrs[cell]-1
      lids = getindex!(cache,cell_to_lposneg,cell)
      cell_owner = lcell_to_owner[cell]
      for (i,lid) in enumerate(lids)
        if (lid > 0) && isequal(lpos_to_owner[lid],cell_owner)
          lpos_to_gpos[lid] = cell_to_gposneg.data[i+p]
        elseif (lid < 0) && isequal(lneg_to_owner[-lid],cell_owner)
          lneg_to_gneg[-lid] = cell_to_gposneg.data[i+p]
        end
      end
    end
  end

  posneg_wise_to_cell_wise!(cell_to_gposneg,lpos_to_gpos,lneg_to_gneg,cell_to_lposneg,own_to_local(cell_range))
  fetch_vector_ghost_values!(cell_to_gposneg,cache_fetch) |> wait
  cell_wise_to_posneg_wise!(lpos_to_gpos,lneg_to_gneg,cell_to_gposneg,cell_to_lposneg,ghost_to_local(cell_range))

  # Setup DoFs LocalIndices
  local_indices_pos = permuted_variable_partition(nopos,lpos_to_gpos,lpos_to_owner;start=first_gpos)
  local_indices_neg = permuted_variable_partition(noneg,lneg_to_gneg,lneg_to_owner;start=first_gneg)

  return PRange(local_indices_pos), PRange(local_indices_neg)
end

"""
    generate_gids_by_color(cell_gids::PRange, cell_to_lids, lid_to_color, ncolors)

Similar to `generate_gids`, but uses a global partition given by `lid_to_color` to generate
a different set of global ids per color. Returns:

- a tuple with a `PRange` of `PermutedLocalIndices` per color.
- a mapping `lid_to_clid` that maps local ids to local color ids
- a mapping `color_to_clid_to_lid` that maps, for each color, local color ids to local ids.
"""
function generate_gids_by_color(
  cell_range::PRange,
  cell_to_lids::AbstractArray{<:AbstractArray},
  lid_to_color::AbstractArray,
  ncolors = getany(reduction(max,map(maximum,lid_to_color);destination=:all))
)
  cell_lids_to_data = allocate_cell_wise_vector(Int, cell_to_lids)
  cache_fetch = fetch_vector_ghost_values_cache(cell_lids_to_data,partition(cell_range))

  lid_to_owner, color_to_noids = map(
    partition(cell_range),cell_to_lids,lid_to_color
  ) do indices, cell_to_lids, lid_to_color
    lid_to_owner = fill(zero(Int32),length(lid_to_color))
    cache = array_cache(cell_to_lids)
    for (cell, owner) in enumerate(local_to_owner(indices))
      lids = getindex!(cache,cell_to_lids,cell)
      for lid in lids
        (lid < 0) && continue
        lid_to_owner[lid] = max(owner,lid_to_owner[lid])
      end
    end

    rank = part_id(indices)
    color_to_noids = zeros(Int,ncolors)
    for (owner, color) in zip(lid_to_owner,lid_to_color)
      color_to_noids[color] += isequal(owner,rank)
    end

    return lid_to_owner, color_to_noids
  end |> tuple_of_arrays

  # Find first gid per color
  color_to_fgid = map(1:ncolors) do c
    scan(+,map(Base.Fix2(getindex,c), color_to_noids); type=:exclusive, init=1)
  end |> to_parray_of_arrays

  # Start exchanging dof owners
  cell_lids_to_owner = dof_wise_to_cell_wise!(
    cell_lids_to_data,lid_to_owner,cell_to_lids,own_to_local(cell_range)
  )
  t1 = fetch_vector_ghost_values!(cell_lids_to_owner,cache_fetch)

  # Note: lid_to_owner is still not consistent, but owned data is correct
  lid_to_gid, lid_to_clid, color_to_clid_to_lid = map(
    partition(cell_range), lid_to_color, lid_to_owner, color_to_fgid
  ) do indices, lid_to_color, lid_to_owner, color_to_fgid

    # Count number of dofs per color
    rank = part_id(indices)
    color_to_nldofs = zeros(Int,ncolors)
    lid_to_clid = zeros(Int32,length(lid_to_color))
    lid_to_gid = zeros(Int,length(lid_to_color))
    for (lid, (color, owner)) in enumerate(zip(lid_to_color,lid_to_owner))
      color_to_nldofs[color] += 1
      lid_to_clid[lid] = color_to_nldofs[color]
      if isequal(owner,rank)
        lid_to_gid[lid] = color_to_fgid[color]
        color_to_fgid[color] += 1
      end
    end

    # Find clid to lid mapping
    color_to_clid_to_lid = [fill(zero(Int32),n) for n in color_to_nldofs]
    for (lid, (color, clid)) in enumerate(zip(lid_to_color,lid_to_clid))
      color_to_clid_to_lid[color][clid] = lid
    end
    return lid_to_gid, lid_to_clid, color_to_clid_to_lid
  end |> tuple_of_arrays

  # Finish exchanging the dof owners.
  wait(t1)
  cell_wise_to_dof_wise!(
    lid_to_owner,cell_lids_to_owner,cell_to_lids,ghost_to_local(cell_range)
  )

  # Start exchanging gids
  cell_to_gids = dof_wise_to_cell_wise!(
    cell_lids_to_data,lid_to_gid,cell_to_lids,own_to_local(cell_range)
  )
  t2 = fetch_vector_ghost_values!(cell_to_gids,cache_fetch)

  # Finish exchanging the gids.
  wait(t2)
  map(
    cell_to_lids,cell_to_gids,lid_to_gid,lid_to_owner,partition(cell_range)
  ) do cell_to_lids,cell_to_gids,lid_to_gid,lid_to_owner,indices
    cache = array_cache(cell_to_lids)
    lcell_to_owner = local_to_owner(indices)
    for cell in ghost_to_local(indices)
      p = cell_to_gids.ptrs[cell]-1
      lids = getindex!(cache,cell_to_lids,cell)
      cell_owner = lcell_to_owner[cell]
      for (i,lid) in enumerate(lids)
        if (lid > 0) && isequal(lid_to_owner[lid],cell_owner)
          lid_to_gid[lid] = cell_to_gids.data[i+p]
        end
      end
    end
  end

  dof_wise_to_cell_wise!(cell_to_gids,lid_to_gid,cell_to_lids,own_to_local(cell_range))
  fetch_vector_ghost_values!(cell_to_gids,cache_fetch) |> wait
  cell_wise_to_dof_wise!(lid_to_gid,cell_to_gids,cell_to_lids,ghost_to_local(cell_range))

  # Setup DoFs indices per color
  color_to_indices = ntuple(ncolors) do c
    c_noids  = map(Base.Fix2(getindex,c), color_to_noids)
    c_fgid   = color_to_fgid[c]
    c_gids   = map(getindex, lid_to_gid,   color_to_clid_to_lid[c])
    c_owners = map(getindex, lid_to_owner, color_to_clid_to_lid[c])
    PRange(permuted_variable_partition(c_noids, c_gids, c_owners; start=c_fgid))
  end

  return color_to_indices, lid_to_clid, color_to_clid_to_lid
end

"""
    split_gids_by_color(gids::PRange, lid_to_color::AbstractArray, ncolors) -> NTuple{ncolors,PRange}

Given a set of global ids and a mapping from local ids to colors, this function splits
the global ids into different sets of global ids per color. Returns a tuple with a `PRange`
of `PermutedLocalIndices` per color.
"""
function split_gids_by_color(
  gids::PRange, lid_to_color::AbstractArray,
  ncolors = getany(reduction(max,map(maximum,lid_to_color);destination=:all))
)
  color_to_nlids, color_to_noids = map(partition(gids), lid_to_color) do ids, lid_to_color
    rank = part_id(ids)
    lid_to_owner = local_to_owner(ids)
    color_to_nlids = zeros(Int,ncolors)
    color_to_noids = zeros(Int,ncolors)
    for (owner,color) in zip(lid_to_owner,lid_to_color)
      color_to_nlids[color] += 1
      color_to_noids[color] += isequal(owner,rank)
    end
    return color_to_nlids, color_to_noids
  end |> tuple_of_arrays

  # I wish we could reduce arrays directly...
  color_to_fgid = map(1:ncolors) do c
    scan(+,map(Base.Fix2(getindex,c), color_to_noids); type=:exclusive, init=1)
  end |> to_parray_of_arrays

  lid_to_cgid = map(partition(gids), lid_to_color, color_to_fgid) do ids, lid_to_color, color_to_fgid
    rank = part_id(ids)
    lid_to_cgid = zeros(Int,length(lid_to_color))
    color_to_offset = zeros(Int,ncolors)
    for (lid,(owner,color)) in enumerate(zip(local_to_owner(ids),lid_to_color))
      if isequal(owner,rank)
        lid_to_cgid[lid] = color_to_fgid[color] + color_to_offset[color]
        color_to_offset[color] += 1
      end
    end
    return lid_to_cgid
  end
  consistent!(PVector(lid_to_cgid,partition(gids))) |> wait

  # Pack per-color gid/owner arrays in a single pass over local dofs
  all_c_gids, all_c_owners = map(
    lid_to_color, lid_to_cgid, partition(gids), color_to_nlids
  ) do lid_to_color, lid_to_cgid, ids, color_to_nlids
    c_gids   = [zeros(Int,   n) for n in color_to_nlids]
    c_owners = [zeros(Int32, n) for n in color_to_nlids]
    offsets = zeros(Int, ncolors)
    for (color, cgid, owner) in zip(lid_to_color, lid_to_cgid, local_to_owner(ids))
      offsets[color] += 1
      lid = offsets[color]
      c_gids[color][lid]   = cgid
      c_owners[color][lid] = owner
    end
    return c_gids, c_owners
  end |> tuple_of_arrays

  return ntuple(ncolors) do c
    c_noids  = map(Base.Fix2(getindex,c), color_to_noids)
    c_fgid   = color_to_fgid[c]
    c_gids   = map(Base.Fix2(getindex,c), all_c_gids)
    c_owners = map(Base.Fix2(getindex,c), all_c_owners)
    PRange(permuted_variable_partition(c_noids, c_gids, c_owners; start=c_fgid))
  end
end

# PRange restriction

function restrict_gids(gids::PRange, new_to_old_lid::AbstractArray)

  n_own = map(partition(gids), new_to_old_lid) do ids, n2o_lid
    rank = part_id(ids)
    return count(isequal(rank), view(local_to_owner(ids), n2o_lid))
  end

  # Assign global ids to owned lids
  first_gid = scan(+,n_own,type=:exclusive,init=one(eltype(n_own)))

  old_lid_to_new_gid = map(first_gid,new_to_old_lid,partition(gids)) do first_gid, n2o_lid, ids
    old_lid_to_new_gid = zeros(Int,local_length(ids))
    old_lid_to_owner = local_to_owner(ids)
    rank = part_id(ids)
    gid = first_gid
    for old in n2o_lid
      if old_lid_to_owner[old] == rank
        old_lid_to_new_gid[old] = gid
        gid += 1
      end
    end
    return old_lid_to_new_gid
  end

  consistent!(PVector(old_lid_to_new_gid,partition(gids))) |> wait

  # Prepare new partition
  n_gids = reduction(+,n_own,destination=:all,init=zero(eltype(n_own)))

  new_indices = map(
    n_gids, old_lid_to_new_gid, new_to_old_lid, partition(gids)
  ) do n_gids, old_lid_to_new_gid, new_to_old_lid, ids
    lid_to_gid   = old_lid_to_new_gid[new_to_old_lid]
    lid_to_owner = local_to_owner(ids)[new_to_old_lid]
    return LocalIndices(n_gids,part_id(ids),lid_to_gid,lid_to_owner)
  end

  p_snd, p_rcv = assembly_neighbors(partition(gids))
  assembly_neighbors(new_indices; neighbors=ExchangeGraph(p_snd, p_rcv))

  return PRange(new_indices)
end

# Multi-field PRange concatenation

"""
    vcat_gids(f_p_flid_lid, f_frange) -> PRange

Concatenate N per-field PRanges into a single multi-field PRange.

- `f_p_flid_lid`: for each field f, a distributed array mapping single-field local ids to multi-field local ids.
- `f_frange`: per-field PRanges.
"""
function vcat_gids(
  f_p_flid_lid::AbstractVector{<:AbstractArray{<:AbstractVector}},
  f_frange::AbstractVector{<:PRange}
)
  f_p_fiset = map(local_views,f_frange)

  v(x...) = collect(x)
  p_f_fiset    = map(v,f_p_fiset...)
  p_f_flid_lid = map(v,f_p_flid_lid...)

  # Per-part own DOF count and first global ID
  p_noids    = map(f_fiset->sum(map(own_length,f_fiset)),p_f_fiset)
  p_firstgid = scan(+,p_noids,type=:exclusive,init=one(eltype(p_noids)))

  # Fill owned gids and owners
  p_lid_gid, p_lid_part = map(p_f_flid_lid, p_f_fiset, p_firstgid) do f_flid_lid, f_fiset, firstgid
    nlids    = sum(length, f_flid_lid)
    lid_gid  = zeros(Int,   nlids)
    lid_part = zeros(Int32, nlids)
    gid = firstgid
    for (flid_lid, fiset) in zip(f_flid_lid, f_fiset)
      part = part_id(fiset)
      for flid in own_to_local(fiset)
        lid_gid[flid_lid[flid]]  = gid
        lid_part[flid_lid[flid]] = part
        gid += 1
      end
    end
    return lid_gid, lid_part
  end |> tuple_of_arrays

  # Propagate gid and owner to ghost DOFs via each field's communicator
  f_aux_gids = map(frange->PVector{Vector{eltype(eltype(p_lid_gid))}}(undef,partition(frange)),f_frange)
  f_aux_part = map(frange->PVector{Vector{eltype(eltype(p_lid_part))}}(undef,partition(frange)),f_frange)
  _vcat_propagate_ghost!(p_lid_gid,f_aux_gids,f_p_flid_lid,f_p_fiset)
  _vcat_propagate_ghost!(p_lid_part,f_aux_part,f_p_flid_lid,f_p_fiset)

  # Build permuted partition
  p_iset = map(p_lid_gid,p_lid_part,p_noids,p_firstgid) do lid_to_gid, lid_to_owner, nodofs, firstgid
    permuted_variable_partition(nodofs,lid_to_gid,lid_to_owner;start=firstgid)
  end

  # Merge assembly neighbors from all fields
  f_p_parts_snd, f_p_parts_rcv = map(assembly_neighbors ∘ partition, f_frange) |> tuple_of_arrays
  merge_neigs(f_neigs...) = sort(unique(vcat(f_neigs...)))
  p_neigs_snd = map(merge_neigs,f_p_parts_snd...)
  p_neigs_rcv = map(merge_neigs,f_p_parts_rcv...)

  exchange_graph = ExchangeGraph(p_neigs_snd,p_neigs_rcv)
  assembly_neighbors(p_iset; neighbors=exchange_graph)

  return PRange(p_iset)
end

function _vcat_propagate_ghost!(
  p_lid_gid, f_gids, f_p_flid_lid, f_p_fiset
)
  for (gids, p_flid_lid, p_fiset) in zip(f_gids, f_p_flid_lid, f_p_fiset)
    p_flid_gid = local_views(gids)
    map(p_flid_gid, p_flid_lid, p_lid_gid, p_fiset) do flid_gid, flid_lid, lid_gid, fiset
      for flid in own_to_local(fiset)
        flid_gid[flid] = lid_gid[flid_lid[flid]]
      end
    end
    cache = fetch_vector_ghost_values_cache(partition(gids), p_fiset)
    fetch_vector_ghost_values!(partition(gids), cache) |> wait
    map(p_flid_gid, p_flid_lid, p_lid_gid, p_fiset) do flid_gid, flid_lid, lid_gid, fiset
      for flid in ghost_to_local(fiset)
        lid_gid[flid_lid[flid]] = flid_gid[flid]
      end
    end
  end
end
