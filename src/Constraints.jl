
function generate_distributed_constraints(
  cell_gids::PRange, spaces::AbstractArray{<:FESpace},
  _sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs
)
  dof_is_slave = map(spaces, _sDOF_to_dof) do space, _sDOF_to_dof
    dof_is_slave = fill(false, num_free_dofs(space))
    dof_is_slave[_sDOF_to_dof] .= true
    return dof_is_slave
  end

  sDOF_gids, mfdof_gids, mddof_gids, sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof = 
    generate_constraint_gids(cell_gids, spaces, dof_is_slave)

  map(sDOF_to_DOF, _sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs) do sDOF_to_DOF, _sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs
    @notimplementedif sDOF_to_DOF != _sDOF_to_dof """
      TODO: Apply permutation to tables
    """
  end

  return generate_distributed_constraints(
    sDOF_gids, mfdof_gids, mddof_gids, 
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
    sDOF_to_mdofs, sDOF_to_coeffs
  )
end

function generate_distributed_constraints(
  cell_gids::PRange, spaces::AbstractArray{<:FESpace}, callback::Function, dof_is_slave
)
  sDOF_gids, mfdof_gids, mddof_gids, sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof = 
    generate_constraint_gids(cell_gids, spaces, dof_is_slave)

  sDOF_to_mdofs, sDOF_to_coeffs = callback(
    sDOF_to_DOF, sDOF_gids
  )

  return generate_distributed_constraints(
    sDOF_gids, mfdof_gids, mddof_gids,
    sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
    sDOF_to_mdofs, sDOF_to_coeffs
  )
end

function generate_distributed_constraints(
  sDOF_gids::PRange, mfdof_gids::PRange, mddof_gids::PRange, 
  sDOF_to_DOF, mfdof_to_DOF, mddof_to_DOF, DOF_to_mDOF, DOF_to_dof,
  sDOF_to_mdofs, sDOF_to_coeffs
)

  # Convert to JaggedArrays 
  _sDOF_to_mdofs = map(sDOF_to_mdofs) do sDOF_to_mdofs
    JaggedArray(sDOF_to_mdofs.data, sDOF_to_mdofs.ptrs)
  end
  _sDOF_to_coeffs = map(sDOF_to_coeffs) do sDOF_to_coeffs
    JaggedArray(sDOF_to_coeffs.data, sDOF_to_coeffs.ptrs)
  end

  # Make tables consistent:
  #  - coeffs are straightforward
  #  - dofs have to be converted to mdof gids, communicated, then converted back to local dofs
  map(to_global_dofs!, _sDOF_to_mdofs, partition(mfdof_gids), partition(mddof_gids), DOF_to_mDOF)
  t1 = consistent!(PVector(_sDOF_to_mdofs, partition(sDOF_gids)))
  t2 = consistent!(PVector(_sDOF_to_coeffs, partition(sDOF_gids)))
  wait(t1)

  mfdof_indices, mddof_indices, mDOF_to_DOF = map(
    to_local_dofs!, _sDOF_to_mdofs, 
    partition(sDOF_gids), partition(mfdof_gids), partition(mddof_gids), 
    mfdof_to_DOF, mddof_to_DOF
  ) |> tuple_of_arrays
  new_mfdof_gids, new_mddof_gids = PRange(mfdof_indices), PRange(mddof_indices)
  wait(t2)

  mDOF_to_dof, sDOF_to_dof = map(
    DOF_to_dof, mDOF_to_DOF, sDOF_to_DOF
  ) do DOF_to_dof, mDOF_to_DOF, sDOF_to_DOF
    for i in eachindex(mDOF_to_DOF)
      iszero(mDOF_to_DOF[i]) && continue
      mDOF_to_DOF[i] = DOF_to_dof[mDOF_to_DOF[i]]
    end
    for i in eachindex(sDOF_to_DOF)
      sDOF_to_DOF[i] = DOF_to_dof[sDOF_to_DOF[i]]
    end
    return mDOF_to_DOF, sDOF_to_DOF
  end |> tuple_of_arrays

  return sDOF_gids, new_mfdof_gids, new_mddof_gids, mDOF_to_dof, sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs
end

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
  cell_gids::PRange, cell_to_DOFs, DOF_is_slave, DOF_to_dof
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

# This is easy, all nonzero entries are local
function to_global_dofs!(sDOF_to_DOFs, mfdof_ids, mddof_ids, DOF_to_mDOF)
  n_lmfdofs = local_length(mfdof_ids)
  n_gmfdofs = global_length(mfdof_ids)
  mfdof_l2g = local_to_global(mfdof_ids)
  mddof_l2g = local_to_global(mddof_ids)
  data = sDOF_to_DOFs.data
  for k in eachindex(data)
    iszero(data[k]) && continue
    mDOF = DOF_to_mDOF[data[k]]
    if mDOF > n_lmfdofs # mddof
      data[k] = mddof_l2g[mDOF - n_lmfdofs] + n_gmfdofs
    else # mfdof
      data[k] = mfdof_l2g[mDOF]
    end
  end
end

# This one is tricky: some nonzero entries will be non-local (i.e roots on other processors).
# We have to add these to the pre-existing dof numbering.
function to_local_dofs!(sDOF_to_mdofs, sDOF_ids, mfdof_gids, mddof_gids, mfdof_to_DOF, mddof_to_DOF)
  rank = part_id(sDOF_ids)
  n_lmfdofs = local_length(mfdof_gids)
  n_lmddofs = local_length(mddof_gids)
  n_gmfdofs = global_length(mfdof_gids)
  new_mfdof = Dict{Int,Tuple{Int32,Int32}}()
  new_mddof = Dict{Int,Tuple{Int32,Int32}}()
  mfdof_g2l = global_to_local(mfdof_gids)
  mddof_g2l = global_to_local(mddof_gids)
  ptrs = sDOF_to_mdofs.ptrs
  data = sDOF_to_mdofs.data
  for (aggdof,owner) in enumerate(local_to_owner(sDOF_ids))
    for k in ptrs[aggdof]:ptrs[aggdof+1]-1
      @assert !iszero(data[k]) # All entries should be nonzero after communication
      gid = data[k]
      if gid <= n_gmfdofs # mfdof
        mdof = mfdof_g2l[gid]
        if iszero(mdof) # Remote mfdof
          mdof, mdof_owner = get!(new_mfdof,gid,(n_lmfdofs+1,owner))
          @assert isequal(mdof_owner,owner) && !isequal(owner, rank)
          n_lmfdofs += isequal(mdof,n_lmfdofs+1) # Only increment if new
        end
      else # mddof
        gid  = gid - n_gmfdofs
        mdof = mddof_g2l[gid]
        if iszero(mdof) # Remote mddof
          mdof, mdof_owner = get!(new_mddof,gid,(n_lmddofs+1,owner))
          @assert isequal(mdof_owner,owner) && !isequal(owner, rank)
          n_lmddofs += isequal(mdof,n_lmddofs+1) # Only increment if new
        end
        mdof = -mdof # Mark as mddof
      end
      data[k] = mdof
    end
  end
  
  # Create expanded master dof numbering
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
    LocalIndices(n_global, rank, lid_to_gid, lid_to_owner)
  end
  new_mfdof_gids = expand_gids(mfdof_gids, new_mfdof)
  new_mddof_gids = expand_gids(mddof_gids, new_mddof)

  mDOF_to_DOF = vcat(
    mfdof_to_DOF, zeros(Int32,length(new_mfdof)),
    mddof_to_DOF, zeros(Int32,length(new_mddof))
  )
  return new_mfdof_gids, new_mddof_gids, mDOF_to_DOF
end

###########################################################################################
###########################################################################################
###########################################################################################


