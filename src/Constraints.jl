#############################
## OPTION 1: Single-constraint set, generated from local spaces and local masters.

# This function REQUIRES that :
#   - all `DOFs` are LOCAL (i.e. no masters on other processors)
#   - The OWNED constraints are filled in, GHOST constraints are allocated (but will be thrown away)
#
# The reason we require this is because we rely on the local space cell_to_dofs to generate
# the global constraint and master gids. 
function generate_distributed_constraints(
  cell_gids::PRange, spaces::AbstractArray{<:FESpace},
  _sDOF_to_dof, sDOF_to_DOFs, sDOF_to_coeffs
)
  dof_is_slave = map(spaces, _sDOF_to_dof) do space, _sDOF_to_dof
    dof_is_slave = fill(false, num_free_dofs(space))
    dof_is_slave[_sDOF_to_dof] .= true
    return dof_is_slave
  end

  sDOF_gids, mfdof_gids, mddof_gids, sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof = 
    generate_constraint_gids(cell_gids, spaces, dof_is_slave)

  map(sDOF_to_DOF, _sDOF_to_dof, sDOF_to_DOFs, sDOF_to_coeffs) do sDOF_to_DOF, _sDOF_to_dof, sDOF_to_DOFs, sDOF_to_coeffs
    @notimplementedif sDOF_to_DOF != _sDOF_to_dof """
      TODO: Apply permutation to tables
    """
  end

  return generate_distributed_constraints(
    sDOF_gids, mfdof_gids, mddof_gids, 
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
    sDOF_to_DOFs, sDOF_to_coeffs
  )
end

# Same as above, but with a callback to generate the master dofs and coefficients. 
# What we callback allows you to do is to wait until after the global constraint gids have been generated.
function generate_distributed_constraints(
  cell_gids::PRange, spaces::AbstractArray{<:FESpace}, callback::Function, dof_is_slave
)
  sDOF_gids, mfdof_gids, mddof_gids, sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof = 
    generate_constraint_gids(cell_gids, spaces, dof_is_slave)

  sDOF_to_DOFs, sDOF_to_coeffs = callback(sDOF_to_DOF, sDOF_gids)

  return generate_distributed_constraints(
    sDOF_gids, mfdof_gids, mddof_gids,
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
    sDOF_to_DOFs, sDOF_to_coeffs
  )
end

function generate_distributed_constraints(
  sDOF_gids::PRange, mfdof_gids::PRange, mddof_gids::PRange,
  sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
  sDOF_to_DOFs, sDOF_to_coeffs
)

  DOF_gids, offsets = concatenate_constraint_gids(
    sDOF_gids, mfdof_gids, mddof_gids,
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF
  )
  new_DOFs = consistent_constraints!(
    sDOF_gids, DOF_gids, sDOF_to_DOFs, sDOF_to_coeffs
  )

  new_mfdof_gids, new_mddof_gids, mDOF_to_dof, sDOF_to_dof = reindex_constraints!(
    sDOF_gids, mfdof_gids, mddof_gids,
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
    sDOF_to_DOFs, new_DOFs, offsets
  )

  return sDOF_gids, new_mfdof_gids, new_mddof_gids, mDOF_to_dof, sDOF_to_dof, sDOF_to_DOFs, sDOF_to_coeffs
end

###########################################################################################
###########################################################################################
###########################################################################################
## OPTION 2: Multiple-constraint sets, generated from generated from local spaces and local masters.
# See `/docs/src/dev/constraints.md` for the rationale behind this implementation.

# In the below implementation, `dof_to_constraint` is an array that, for each local dof, 
# gives the constraint set it belongs to (0 for unconstrained dofs).
# In particular we have `dof_is_slave = dof_to_constraint .> 0`.
#
# This also means that only ONE constraint can be applied to each DOF 
# (i.e. we cannot arbitrarily combine constraints from different sets on the same DOF).
# Also, this means that, for each constraint set, 
#    `` c_callback(csDOF_to_DOF, csDOF_gids) ``
# might not asked for ALL possible slaves in that set, but only a subset of them
# (e.g. only the ones that have been selected). The callbacks need to account for that.
function generate_distributed_constraints(
  cell_gids::PRange, spaces::AbstractArray{<:FESpace}, callback::Tuple, dof_to_constraint
)

  # Create constraint gids
  dof_is_slave = map(x -> x .> 0, dof_to_constraint)
  sDOF_gids, mfdof_gids, mddof_gids, sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof = 
    generate_constraint_gids(cell_gids, spaces, dof_is_slave)

  # Generate partial constraints and merge them into (inconsistent but complete) slave tables 
  nc = length(callback)
  sDOF_to_c = map(getindex, dof_to_constraint, sDOF_to_DOF)
  c_to_csDOF_gids, sDOF_to_csDOF = split_gids_by_color(sDOF_gids, sDOF_to_c, nc)

  c_to_csDOF_to_DOF = map(sDOF_to_DOF, sDOF_to_c) do sDOF_to_DOF, sDOF_to_c
    ntuple(nc) do c
      sDOF_to_DOF[findall(==(c), sDOF_to_c)]
    end
  end |> tuple_of_arrays

  partials = map(x -> Vector{Any}(undef, nc), partition(cell_gids))
  for (c, (cb, csDOF_to_DOF, csDOF_gids)) in enumerate(zip(callback, c_to_csDOF_to_DOF, c_to_csDOF_gids))
    csDOF_to_DOFs, csDOF_to_coeffs = cb(csDOF_to_DOF, csDOF_gids)
    map(partials, csDOF_to_DOFs, csDOF_to_coeffs) do partials, csDOF_to_DOFs, csDOF_to_coeffs
      partials[c] = (csDOF_to_DOFs, csDOF_to_coeffs)
    end
  end

  sDOF_to_DOFs, sDOF_to_coeffs = map(sDOF_to_c,sDOF_to_csDOF,partials) do sDOF_to_c, sDOF_to_csDOF, partials
    c_to_csDOF_DOFs = map(first, partials)
    sDOF_to_DOFs = Arrays.merge_entries(sDOF_to_c, sDOF_to_csDOF, c_to_csDOF_DOFs...)
    c_to_csDOF_coeffs = map(last, partials)
    sDOF_to_coeffs = Arrays.merge_entries(sDOF_to_c, sDOF_to_csDOF, c_to_csDOF_coeffs...)
    return sDOF_to_DOFs, sDOF_to_coeffs
  end |> tuple_of_arrays

  # Make constraints consistent
  DOF_gids, offsets = concatenate_constraint_gids(
    sDOF_gids, mfdof_gids, mddof_gids,
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF
  )
  new_DOFs = consistent_constraints!(
    sDOF_gids, DOF_gids, sDOF_to_DOFs, sDOF_to_coeffs
  )

  # Close fully consistent constraint tables
  sDOF_indices, sDOF_to_DOF, sDOF_to_DOFs, sDOF_to_coeffs = map(
    spaces, new_DOFs, sDOF_to_DOF, sDOF_to_DOFs, sDOF_to_coeffs, partition(sDOF_gids)
  ) do space, new_DOFs, sDOF_to_DOF, sDOF_to_DOFs, sDOF_to_coeffs, sDOF_ids
    n_DOFs = num_free_dofs(space) + num_dirichlet_dofs(space) + length(new_DOFs)
    new_sDOF_to_DOF, new_sDOF_to_DOFs, new_sDOF_to_coeffs, _ = FESpaces.close_slave_constraint_tables(
      n_DOFs, n_DOFs, sDOF_to_DOF, sDOF_to_DOFs, sDOF_to_coeffs; keys = local_to_global(sDOF_ids)
    )

    # close! reindexes the constraints locally to follow the topological ordering of the DAG.
    # We need to apply the same permutation to the sDOF_gids to keep consistency.
    DOF_to_sDOF = find_inverse_index_map(sDOF_to_DOF, n_DOFs)
    perm = DOF_to_sDOF[new_sDOF_to_DOF]
    new_sDOF_ids = LocalIndices(
      global_length(sDOF_ids), part_id(sDOF_ids), 
      local_to_global(sDOF_ids)[perm], local_to_owner(sDOF_ids)[perm]
    )

    return new_sDOF_ids, new_sDOF_to_DOF, new_sDOF_to_DOFs, new_sDOF_to_coeffs
  end |> tuple_of_arrays
  sDOF_gids = PRange(sDOF_indices)

  # Reindex DOF tables to signed mDOF indices, expand mfdof/mddof gids, etc..
  new_mfdof_gids, new_mddof_gids, mDOF_to_dof, sDOF_to_dof = reindex_constraints!(
    sDOF_gids, mfdof_gids, mddof_gids,
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
    sDOF_to_DOFs, new_DOFs, offsets
  )
  sDOF_to_mdofs = sDOF_to_DOFs # Has been reindexed!

  return sDOF_gids, new_mfdof_gids, new_mddof_gids, mDOF_to_dof, sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs
end

###########################################################################################
###########################################################################################
###########################################################################################
# Helpers

function generate_constraint_gids(
  cell_gids::PRange, spaces::AbstractArray{<:FESpace}, dof_is_slave
)
  cell_to_DOFs, DOF_is_slave, DOF_to_dof = map(spaces, dof_is_slave) do space, dof_is_slave
    nfree = num_free_dofs(space)
    ndir  = num_dirichlet_dofs(space)
    if iszero(ndir)
      nlDOFs = nfree
      cell_to_DOFs = get_cell_dof_ids(space)
      DOF_is_slave = dof_is_slave
      DOF_to_dof = collect(Int32(1):Int32(nlDOFs))
    else
      nlDOFs = nfree + ndir
      dof_reindex = PosNegReindex(Int32(1):Int32(nfree),Int32(nfree+1):Int32(nlDOFs))
      cell_to_DOFs = lazy_map(Broadcasting(dof_reindex),get_cell_dof_ids(space))
      DOF_is_slave = vcat(dof_is_slave, zeros(eltype(dof_is_slave),ndir))
      DOF_to_dof = vcat(Int32(1):Int32(nfree),-(Int32(1):Int32(ndir)))
    end
    return cell_to_DOFs, DOF_is_slave, DOF_to_dof
  end |> tuple_of_arrays

  return generate_constraint_gids(
    cell_gids, cell_to_DOFs, DOF_is_slave, DOF_to_dof
  )
end

function generate_constraint_gids(
  cell_gids::PRange, cell_to_DOFs::AbstractArray, DOF_is_slave, DOF_to_dof
)
  # Create pos/neg local numberings
  DOF_to_color = map(DOF_is_slave, DOF_to_dof) do DOF_is_slave, DOF_to_dof
    DOF_to_color = zeros(Int8,length(DOF_is_slave))
    for DOF in eachindex(DOF_is_slave)
      if !iszero(DOF_is_slave[DOF]) # Slave DOF (1)
        DOF_to_color[DOF] = Int8(1)
      else # Master DOF: free (2) / dirichlet (3)
        DOF_to_color[DOF] = Int8(2) + Int8(DOF_to_dof[DOF] < 1)
      end
    end
    return DOF_to_color
  end

  # Generate global master and slave dof ids
  gids, DOF_to_clid, color_to_clid_to_lid = generate_gids_by_color(
    cell_gids, cell_to_DOFs, DOF_to_color, 3
  )
  sDOF_gids, mfdof_gids, mddof_gids = gids

  sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF = map(
    DOF_to_clid, color_to_clid_to_lid
  ) do DOF_to_clid, color_to_clid_to_lid
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF = color_to_clid_to_lid
    n_mfdofs = length(mfdof_to_DOF)
    DOF_to_mDOF = zeros(Int32,length(DOF_to_clid))
    for (mfdof, DOF) in enumerate(mfdof_to_DOF)
      DOF_to_mDOF[DOF] = mfdof
    end
    for (mddof, DOF) in enumerate(mddof_to_DOF)
      DOF_to_mDOF[DOF] = n_mfdofs + mddof
    end
    return sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF
  end |> tuple_of_arrays

  return sDOF_gids, mfdof_gids, mddof_gids, sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof
end

function consistent_constraints!(
  sDOF_gids::PRange, DOF_gids::PRange, sDOF_to_DOFs, sDOF_to_coeffs
)

  # Map to global dofs
  map(sDOF_to_DOFs, partition(DOF_gids)) do sDOF_to_DOFs, DOF_ids
    l2g = local_to_global(DOF_ids)
    data = sDOF_to_DOFs.data
    for k in eachindex(data)
      iszero(data[k]) && continue
      data[k] = l2g[data[k]]
    end
  end

  t1 = consistent!(PVector(map(jagged_array,sDOF_to_DOFs), partition(sDOF_gids)))
  t2 = consistent!(PVector(map(jagged_array,sDOF_to_coeffs), partition(sDOF_gids)))
  wait(t1)

  # Map to local dofs, gather external DOFs
  new_DOFs = map(sDOF_to_DOFs, partition(DOF_gids), partition(sDOF_gids)) do sDOF_to_DOFs, DOF_ids, sDOF_ids
    rank = part_id(DOF_ids)
    n_DOF = length(DOF_ids)
    ptrs = sDOF_to_DOFs.ptrs
    data = sDOF_to_DOFs.data
    g2l = global_to_local(DOF_ids)
    new_DOFs = Dict{Int,Tuple{Int32,Int32}}()
    for (sdof,owner) in enumerate(local_to_owner(sDOF_ids))
      for k in ptrs[sdof]:ptrs[sdof+1]-1
        @assert !iszero(data[k]) # All entries should be nonzero after communication
        gid = data[k]
        lid = g2l[gid]
        if iszero(lid) # Remote DOF
          lid, dof_owner = get!(new_DOFs,gid,(n_DOF+1,owner))
          @assert isequal(dof_owner, owner) && !isequal(owner, rank)
          n_DOF += isequal(lid,n_DOF+1) # Only increment if new
        end
        data[k] = lid
      end
    end
    return new_DOFs
  end

  wait(t2)
  return new_DOFs
end

# Reindexes the master DOF entries in sDOF_to_DOFs (currently holding extended local DOF
# indices after consistent_constraints!) to signed local mDOF indices (positive = mfdof,
# negative = mddof). Also expands mfdof_gids and mddof_gids to include any fictitious
# master DOFs that arrived from other processors, and produces mDOF_to_dof / sDOF_to_dof.
#
# offsets = cumsum((0, global_length(sDOF_gids), global_length(mfdof_gids), global_length(mddof_gids)))
# new_DOFs: per-part Dict{Int,Tuple{Int32,Int32}} mapping DOF_global_gid → (DOF_lid, owner),
#           built by consistent_constraints!.
function reindex_constraints!(
  sDOF_gids::PRange, mfdof_gids::PRange, mddof_gids::PRange,
  sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
  sDOF_to_DOFs, new_DOFs, offsets
)
  new_mfdof_indices, new_mddof_indices, mDOF_to_dof, sDOF_to_dof = map(
    partition(mfdof_gids), partition(mddof_gids),
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
    sDOF_to_DOFs, new_DOFs
  ) do mfdof_ids, mddof_ids,
      sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
      sDOF_to_DOFs, new_DOFs

    n_DOFs   = length(DOF_to_mDOF)
    n_mfdofs = length(mfdof_to_DOF)
    n_mddofs = length(mddof_to_DOF)

    # Classify each fictitious DOF as mfdof or mddof using its global DOF gid,
    # assign it a new local mDOF lid, and record the DOF_lid → signed_mDOF_lid
    # mapping needed for the sDOF_to_DOFs conversion below.
    new_mfdof = Dict{Int,Tuple{Int32,Int32}}()
    new_mddof = Dict{Int,Tuple{Int32,Int32}}()
    DOF_lid_to_signed_mDOF = Dict{Int,Int32}()
    n_new_mfdofs = 0
    n_new_mddofs = 0
    for (DOF_gid, (DOF_lid, owner)) in new_DOFs
      if DOF_gid <= offsets[3]  # mfdof: offsets[2] < DOF_gid <= offsets[3]
        n_new_mfdofs += 1
        mfdof_gid = DOF_gid - offsets[2]
        mfdof_lid = Int32(n_mfdofs + n_new_mfdofs)
        new_mfdof[mfdof_gid] = (mfdof_lid, owner)
        DOF_lid_to_signed_mDOF[DOF_lid] = mfdof_lid
      else  # mddof: offsets[3] < DOF_gid <= offsets[4]
        n_new_mddofs += 1
        mddof_gid = DOF_gid - offsets[3]
        mddof_lid = Int32(n_mddofs + n_new_mddofs)
        new_mddof[mddof_gid] = (mddof_lid, owner)
        DOF_lid_to_signed_mDOF[DOF_lid] = -mddof_lid
      end
    end

    # Reindex sDOF_to_DOFs.data in-place: local DOF indices → signed local mDOF indices
    data = sDOF_to_DOFs.data
    for k in eachindex(data)
      DOF = data[k]
      if DOF <= n_DOFs
        mDOF = DOF_to_mDOF[DOF]
        data[k] = (mDOF <= n_mfdofs) ? mDOF : -(mDOF - n_mfdofs)
      else  # fictitious DOF introduced by consistent_constraints!
        data[k] = DOF_lid_to_signed_mDOF[DOF]
      end
    end

    # mDOF_to_dof layout: [orig mfdofs | new fictitious mfdofs (0) | orig mddofs | new fictitious mddofs (0)]
    mDOF_to_dof = zeros(Int32, n_mfdofs + n_new_mfdofs + n_mddofs + n_new_mddofs)
    for (mfdof, DOF) in enumerate(mfdof_to_DOF)
      mDOF_to_dof[mfdof] = DOF_to_dof[DOF]
    end
    mddof_offset = n_mfdofs + n_new_mfdofs
    for (mddof, DOF) in enumerate(mddof_to_DOF)
      mDOF_to_dof[mddof_offset + mddof] = DOF_to_dof[DOF]
    end

    sDOF_to_dof = DOF_to_dof[sDOF_to_DOF]

    new_mfdof_ids = expand_gids(mfdof_ids, new_mfdof)
    new_mddof_ids = expand_gids(mddof_ids, new_mddof)
    return new_mfdof_ids, new_mddof_ids, mDOF_to_dof, sDOF_to_dof
  end |> tuple_of_arrays

  new_mfdof_gids = PRange(new_mfdof_indices)
  new_mddof_gids = PRange(new_mddof_indices)
  return new_mfdof_gids, new_mddof_gids, mDOF_to_dof, sDOF_to_dof
end

# Can be replaced by union_ghost in PartitionedArrays v0.5
function expand_gids(gids, new_gids)
  rank = part_id(gids)
  n_global = global_length(gids)
  n_old = local_length(gids)
  n_new = n_old + length(new_gids)
  lid_to_gid = Vector{Int}(undef,n_new)
  lid_to_owner = Vector{Int32}(undef,n_new)
  lid_to_gid[1:n_old] .= local_to_global(gids)
  lid_to_owner[1:n_old] .= local_to_owner(gids)
  for (gid, (lid, owner)) in new_gids
    lid_to_gid[lid] = gid
    lid_to_owner[lid] = owner
  end
  return LocalIndices(n_global, rank, lid_to_gid, lid_to_owner)
end

function concatenate_constraint_gids(
  sDOF_gids::PRange, mfdof_gids::PRange, mddof_gids::PRange,
  sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF
)
  function map_b_to_a!(a_l2g,a_l2o,b2a,b_ids,o)
    @views a_l2g[b2a] .= local_to_global(b_ids) .+ o
    @views a_l2o[b2a] .= local_to_owner(b_ids)
  end
  offsets = cumsum((0, map(length, (sDOF_gids, mfdof_gids, mddof_gids))...))
  DOF_gids_indices = map(
    partition(sDOF_gids), partition(mfdof_gids), partition(mddof_gids),
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF
  ) do s_ids, mf_ids, md_ids, sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF
    n_DOFs = length(sDOF_to_DOF) + length(mfdof_to_DOF) + length(mddof_to_DOF)
    lid_to_gid   = Vector{Int}(undef, n_DOFs)
    lid_to_owner = Vector{Int32}(undef, n_DOFs)
    map_b_to_a!(lid_to_gid, lid_to_owner, sDOF_to_DOF, s_ids, offsets[1])
    map_b_to_a!(lid_to_gid, lid_to_owner, mfdof_to_DOF, mf_ids, offsets[2])
    map_b_to_a!(lid_to_gid, lid_to_owner, mddof_to_DOF, md_ids, offsets[3])
    LocalIndices(offsets[end], part_id(s_ids), lid_to_gid, lid_to_owner)
  end
  return PRange(DOF_gids_indices), offsets
end

###########################################################################################
###########################################################################################
###########################################################################################
## OPTION 3: Single-constraint set, generated from global space and (possibly) non-local masters.
# TODO: I am not 100% sure how to make this useful. In what kind of situations would 
#       we have non-local masters but no global numbering? It is weird.

# This function allows for non-local masters, but REQUIRES a pre-existing global numbering 
# for the original dofs.
# function generate_constraint_gids(
#   DOF_gids::PRange, sDOF_to_DOF::AbstractArray{<:AbstractVector}, sDOF_to_DOFs, DOF_to_isdirichlet
# )
#   DOF_to_color = map(
#     partition(DOF_gids), sDOF_to_DOF, sDOF_to_DOFs, DOF_to_isdirichlet
#   ) do DOF_ids, sDOF_to_DOF, sDOF_to_DOFs, DOF_to_isdirichlet
#     DOF_to_color = zeros(Int8, length(DOF_ids))
#     DOF_to_oDOF = local_to_own(DOF_ids)
#     for (sDOF, DOF) in enumerate(sDOF_to_DOF)
#       iszero(DOF_to_oDOF[DOF]) && continue # Ghost DOF, ignore
#       DOF_to_color[DOF] = 1
#       for DOF2 in dataview(sDOF_to_DOFs, sDOF)
#         DOF_to_color[DOF2] = 2 + Int8(DOF_to_isdirichlet[DOF2])
#       end
#     end
#     return DOF_to_color
#   end
#   wait(consistent!(PVector(DOF_to_color, partition(DOF_gids))))
# 
#   sDOF_gids, mfdof_gids, mddof_gids = split_gids_by_color(
#     DOF_gids, DOF_to_color, 3
#   )
# 
#   mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF = map(
#     DOF_to_color
#   ) do DOF_to_clid, color_to_clid_to_lid
#     sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF = color_to_clid_to_lid
#     n_mfdofs = length(mfdof_to_DOF)
#     DOF_to_mDOF = zeros(Int32,length(DOF_to_clid))
#     for (mfdof, DOF) in enumerate(mfdof_to_DOF)
#       DOF_to_mDOF[DOF] = mfdof
#     end
#     for (mddof, DOF) in enumerate(mddof_to_DOF)
#       DOF_to_mDOF[DOF] = n_mfdofs + mddof
#     end
#     return sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF
#   end |> tuple_of_arrays
# 
#   return sDOF_gids, mfdof_gids, mddof_gids, sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof
# end
