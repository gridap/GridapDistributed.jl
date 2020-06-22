struct DistributedFESpace{V} <: FESpace
  vector_type::Type{V}
  spaces::DistributedData{<:FESpace}
  gids::DistributedIndexSet
end

function get_distributed_data(dspace::DistributedFESpace)
  spaces = dspace.spaces
  gids = dspace.gids

  DistributedData(spaces,gids) do part, space, lgids
    space, lgids
  end
end

# Minimal FE interface

function Gridap.FESpaces.num_free_dofs(f::DistributedFESpace)
  f.gids.ngids
end

function Gridap.FESpaces.FEFunction(dV::DistributedFESpace,x)
  dfree_vals = x[dV.gids]
  # IMPORTANT NOTE: we need to call collect below in order to duplicate the
  #                 local portion of dfree_vals. When dfree_vals is of
  #                 type MPIPETScDistributedVector, the Julia's GC can destroy
  #                 the vector on which the entries of dfree_vals are ultimately
  #                 stored when it goes out of scope.
  funs = DistributedData(dV.spaces,dfree_vals) do part, V, free_vals
    FEFunction(V,collect(free_vals))
  end
  DistributedFEFunction(funs,x,dV)
end

function Gridap.FESpaces.EvaluationFunction(dV::DistributedFESpace,x)
  dfree_vals = x[dV.gids]
  # IMPORTANT NOTE: we need to call collect below in order to duplicate the
  #                 local portion of dfree_vals. When dfree_vals is of
  #                 type MPIPETScDistributedVector, the Julia's GC can destroy
  #                 the vector on which the entries of dfree_vals are ultimately
  #                 stored when it goes out of scope.
  funs = DistributedData(dV.spaces,dfree_vals) do part, V, free_vals
    Gridap.FESpaces.EvaluationFunction(V,collect(free_vals))
  end
  DistributedFEFunction(funs,x,dV)
end

function Gridap.FESpaces.zero_free_values(f::DistributedFESpace)
  fv = Gridap.Algebra.allocate_vector(f.vector_type,f.gids)
  fill_entries!(fv,zero(eltype(fv)))
  fv
end

function Gridap.FESpaces.get_cell_basis(f::DistributedFESpace)
  bases = DistributedData(f.spaces) do part, space
    get_cell_basis(space)
  end
  DistributedCellBasis(bases)
end

# FE Function

struct DistributedFEFunction{T}
  funs::DistributedData
  vals::T #::AbstractVector
  space::DistributedFESpace
end

Gridap.FESpaces.FEFunctionStyle(::Type{DistributedFEFunction}) = Val{true}()

get_distributed_data(u::DistributedFEFunction) = u.funs

Gridap.FESpaces.get_free_values(a::DistributedFEFunction) = a.vals

Gridap.FESpaces.get_fe_space(a::DistributedFEFunction) = a.space

Gridap.FESpaces.is_a_fe_function(a::DistributedFEFunction) = true

# Cell basis

struct DistributedCellBasis
  bases::DistributedData
end

Gridap.FESpaces.FECellBasisStyle(::Type{DistributedCellBasis}) = Val{true}()

get_distributed_data(u::DistributedCellBasis) = u.bases

#  Constructors

function Gridap.TrialFESpace(V::DistributedFESpace,args...)
  spaces = DistributedData(V.spaces) do part, space
    TrialFESpace(space,args...)
  end
  DistributedFESpace(V.vector_type,spaces,V.gids)
end

function Gridap.FESpace(::Type{V};model::DistributedDiscreteModel,kwargs...) where V
  DistributedFESpace(V;model=model,kwargs...)
end

function DistributedFESpace(::Type{V}; model::DistributedDiscreteModel,kwargs...) where V

  comm = get_comm(model)

  nsubdoms = num_parts(model.models)

  function init_local_spaces(part,model)
    lspace = FESpace(;model=model,kwargs...)
  end

  spaces = DistributedData(init_local_spaces,comm,model.models)

  function init_lid_to_owner(part,lspace,cell_gids)
    nlids = num_free_dofs(lspace)
    lid_to_owner = zeros(Int,nlids)
    cell_to_part = cell_gids.lid_to_owner
    cell_to_lids = Table(get_cell_dofs(lspace))
    _fill_max_part_around!(lid_to_owner,cell_to_part,cell_to_lids)
    lid_to_owner
  end

  part_to_lid_to_owner = DistributedData{Vector{Int}}(init_lid_to_owner,comm,spaces,model.gids)

  function count_owned_lids(part,lid_to_owner)
    count(owner -> owner == part,lid_to_owner)
  end

  a = DistributedData{Int}(count_owned_lids,comm,part_to_lid_to_owner)
  part_to_num_oids = gather(a)

  if i_am_master(comm)
    ngids = sum(part_to_num_oids)
    _fill_offsets!(part_to_num_oids)
  else
    ngids = -1
  end

  offsets = scatter(comm,part_to_num_oids)
  part_to_ngids = scatter_value(comm,ngids)

  do_on_parts(comm,part_to_ngids) do part, lngids
      ngids=lngids
  end

  num_dofs_x_cell = 0
  do_on_parts(comm,spaces) do part, lspace
    cell_dofs=get_cell_dofs(lspace)
    num_dofs_x_cell=length(cell_dofs[1])
  end

  function init_cell_to_owners(part,cell_to_owners,lspace,lid_to_owner)
    cell_to_lids = get_cell_dofs(lspace)
    dlid_to_zero = zeros(eltype(lid_to_owner),num_dirichlet_dofs(lspace))
    cell_to_owners_from = LocalToGlobalPosNegArray(cell_to_lids,lid_to_owner,dlid_to_zero)
    for i=1:length(cell_to_owners_from)
      for j=1:length(cell_to_owners_from[i])
        cell_to_owners[i][j]=cell_to_owners_from[i][j]
      end
    end
  end

  part_to_cell_to_owners = DistributedVector{Vector{Int}}(model.gids, num_dofs_x_cell)

  do_on_parts(init_cell_to_owners,part_to_cell_to_owners,spaces,part_to_lid_to_owner)

  exchange!(part_to_cell_to_owners)

  function update_lid_to_owner(part,lid_to_owner,lspace,cell_to_owners)
    cell_to_lids = Table(get_cell_dofs(lspace))
    _update_lid_to_owner!(lid_to_owner,cell_to_lids,cell_to_owners)
  end

  do_on_parts(update_lid_to_owner,part_to_lid_to_owner,spaces,part_to_cell_to_owners)

  function init_lid_to_gids(part,lid_to_owner,offset)
    lid_to_gid = zeros(Int,length(lid_to_owner))
    _fill_owned_gids!(lid_to_gid,lid_to_owner,part,offset)
    lid_to_gid
  end

  part_to_lid_to_gid = DistributedData{Vector{Int}}(
    init_lid_to_gids,comm,part_to_lid_to_owner,offsets)

  part_to_cell_to_gids = DistributedVector{Vector{Int}}(
      model.gids,num_dofs_x_cell)

  do_on_parts(init_cell_to_owners,part_to_cell_to_gids,spaces,part_to_lid_to_gid)

  exchange!(part_to_cell_to_gids)

  function update_lid_to_gid(part,lid_to_gid,lid_to_owner,lspace,cell_to_gids,cell_gids)
    cell_to_lids = Table(get_cell_dofs(lspace))
    cell_to_owner = cell_gids.lid_to_owner
    _update_lid_to_gid!(
      lid_to_gid,cell_to_lids,cell_to_gids,cell_to_owner,lid_to_owner)
  end

  do_on_parts(
    update_lid_to_gid,part_to_lid_to_gid,part_to_lid_to_owner,spaces,part_to_cell_to_gids,model.gids)

  exchange!(part_to_cell_to_gids)

  do_on_parts(update_lid_to_owner,part_to_lid_to_gid,spaces,part_to_cell_to_gids)

  function init_free_gids(part,lid_to_gid,lid_to_owner,ngids)
    IndexSet(ngids,lid_to_gid,lid_to_owner)
  end

  gids = DistributedIndexSet(init_free_gids,comm,ngids, part_to_lid_to_gid,part_to_lid_to_owner,part_to_ngids)

  DistributedFESpace(V,spaces,gids)
end

function _update_lid_to_gid!(lid_to_gid,cell_to_lids,cell_to_gids,cell_to_owner,lid_to_owner)
  for cell in 1:length(cell_to_lids)
    i_to_gid = cell_to_gids[cell]
    pini = cell_to_lids.ptrs[cell]
    pend = cell_to_lids.ptrs[cell+1]-1
    cellowner = cell_to_owner[cell]
    for (i,p) in enumerate(pini:pend)
      lid = cell_to_lids.data[p]
      if lid > 0
        owner = lid_to_owner[lid]
        if owner == cellowner
          gid = i_to_gid[i]
          lid_to_gid[lid] = gid
        end
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
      if lid > 0
        owner = i_to_owner[i]
        lid_to_owner[lid] = owner
      end
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
