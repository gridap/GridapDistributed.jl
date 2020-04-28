struct DistributedFESpace
  spaces::ScatteredVector{<:FESpace}
  gids::GhostedVector{Int}
end

function get_spaces_and_gids(dspace::DistributedFESpace)
  spaces = dspace.spaces
  gids = dspace.gids
  comm = get_comm(spaces)
  nparts = num_parts(spaces)

  T = Tuple{get_part_type(spaces),get_part_type(gids)}
  ScatteredVector{T}(comm,nparts,spaces,gids) do part, space, lgids
    space, lgids
  end
end

function Gridap.FESpace(comm::Communicator;model::DistributedDiscreteModel,kwargs...)
  DistributedFESpace(comm;model=model,kwargs...)
end

function DistributedFESpace(comm::Communicator;model::DistributedDiscreteModel,kwargs...)

  nsubdoms = num_parts(model.models)

  function init_local_spaces(part,model)
    lspace = FESpace(;model=model,kwargs...)
  end

  spaces = ScatteredVector{FESpace}(init_local_spaces,comm,nsubdoms,model.models)

  function init_lid_to_owner(part,lspace,cell_gids)
    nlids = num_free_dofs(lspace)
    lid_to_owner = zeros(Int,nlids)
    cell_to_part = cell_gids.lid_to_owner
    cell_to_lids = Table(get_cell_dofs(lspace))
    _fill_max_part_around!(lid_to_owner,cell_to_part,cell_to_lids)
    lid_to_owner
  end

  part_to_lid_to_owner = ScatteredVector{Vector{Int}}(init_lid_to_owner,comm,nsubdoms,spaces,model.gids)

  function count_owned_lids(part,lid_to_owner)
    count(owner -> owner == part,lid_to_owner)
  end

  a = ScatteredVector{Int}(count_owned_lids,comm,nsubdoms,part_to_lid_to_owner)
  part_to_num_oids = gather(a)

  if i_am_master(comm)
    ngids = sum(part_to_num_oids)
    ngids_array = fill(ngids,nsubdoms)
    _fill_offsets!(part_to_num_oids)
  else
    ngids_array = Int[]
  end

  part_to_ngids = scatter(comm,ngids_array)

  offsets = scatter(comm,part_to_num_oids)

  function init_cell_to_owners(part,lspace,lid_to_owner)
    cell_to_lids = get_cell_dofs(lspace)
    dlid_to_zero = zeros(eltype(lid_to_owner),num_dirichlet_dofs(lspace))
    cell_to_owners = collect(LocalToGlobalPosNegArray(cell_to_lids,lid_to_owner,dlid_to_zero))
    cell_to_owners
  end

  part_to_cell_to_owners = GhostedVector{Vector{Int}}(init_cell_to_owners,model.gids,spaces,part_to_lid_to_owner)

  exchange!(part_to_cell_to_owners)

  function update_lid_to_owner(part,lid_to_owner,lspace,cell_to_owners)
    cell_to_lids = Table(get_cell_dofs(lspace))
    _update_lid_to_owner!(lid_to_owner,cell_to_lids,cell_to_owners.lid_to_item)
  end

  do_on_parts(update_lid_to_owner,part_to_lid_to_owner,spaces,part_to_cell_to_owners)

  function init_lid_to_gids(part,lid_to_owner,offset)
    lid_to_gid = zeros(Int,length(lid_to_owner))
    _fill_owned_gids!(lid_to_gid,lid_to_owner,part,offset)
    lid_to_gid
  end

  part_to_lid_to_gid = ScatteredVector{Vector{Int}}(init_lid_to_gids,comm,nsubdoms,part_to_lid_to_owner,offsets)

  part_to_cell_to_gids = GhostedVector{Vector{Int}}(init_cell_to_owners,model.gids,spaces,part_to_lid_to_gid)

  exchange!(part_to_cell_to_gids)

  function update_lid_to_gid(part,lid_to_gid,lid_to_owner,lspace,cell_to_gids)
    cell_to_lids = Table(get_cell_dofs(lspace))
    cell_to_owner = cell_to_gids.lid_to_owner
    _update_lid_to_gid!(lid_to_gid,cell_to_lids,cell_to_gids.lid_to_item,cell_to_owner,lid_to_owner)
  end

  do_on_parts(update_lid_to_gid,part_to_lid_to_gid,part_to_lid_to_owner,spaces,part_to_cell_to_gids)

  exchange!(part_to_cell_to_gids)

  do_on_parts(update_lid_to_owner,part_to_lid_to_gid,spaces,part_to_cell_to_gids)

  function init_free_gids(part,lid_to_gid,lid_to_owner,ngids)
    GhostedVectorPart(ngids,lid_to_gid,lid_to_gid,lid_to_owner)
  end

  gids = GhostedVector{Int}(init_free_gids,comm,nsubdoms,part_to_lid_to_gid,part_to_lid_to_owner,part_to_ngids)

  DistributedFESpace(spaces,gids)
end

function _update_lid_to_gid!(lid_to_gid,cell_to_lids,cell_to_gids,cell_to_owner,lid_to_owner)
  for cell in 1:length(cell_to_lids)
    i_to_gid = cell_to_gids[cell]
    pini = cell_to_lids.ptrs[cell]
    pend = cell_to_lids.ptrs[cell+1]-1
    cellowner = cell_to_owner[cell]
    for (i,p) in enumerate(pini:pend)
      lid = cell_to_lids.data[p]
      owner = lid_to_owner[lid]
      if owner == cellowner
        gid = i_to_gid[i]
        lid_to_gid[lid] = gid
      end
    end
  end
end

function _update_lid_to_owner!(lid_to_owner,cell_to_lids,cell_to_owners)
  for cell in 1:length(cell_to_lids)
    i_to_owner = cell_to_owners[cell]
    pini = cell_to_lids.ptrs[cell]
    pend = cell_to_lids.ptrs[cell+1]-1
    for (i,p) in enumerate(pini:pend)
      lid = cell_to_lids.data[p]
      owner = i_to_owner[i]
      lid_to_owner[lid] = owner
    end
  end
end

function _fill_owned_gids!(lid_to_gid,lid_to_owner,part,offset)
  o = offset
  for (lid,owner) in enumerate(lid_to_owner)
    if owner == part
      o += 1
      lid_to_gid[lid] = o
    end
  end
end

function _fill_offsets!(part_to_num_oids)
  o = 0
  for part in 1:length(part_to_num_oids)
    a = part_to_num_oids[part]
    part_to_num_oids[part] = o
    o += a
  end
end

function _fill_max_part_around!(lid_to_owner,cell_to_owner,cell_to_lids)
  for cell in 1:length(cell_to_lids)
    cellowner = cell_to_owner[cell]
    pini = cell_to_lids.ptrs[cell]
    pend = cell_to_lids.ptrs[cell+1]-1
    for p in pini:pend
      lid = cell_to_lids.data[p]
      if lid > 0
        owner = lid_to_owner[lid]
        lid_to_owner[lid] = max(owner,cellowner)
      end
    end
  end
end


