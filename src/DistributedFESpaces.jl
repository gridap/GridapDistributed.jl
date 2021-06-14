
abstract type DistributedFESpace <: FESpace end

function get_distributed_data(dspace::DistributedFESpace)
    spaces = dspace.spaces
    gids = dspace.gids

    DistributedData(spaces, gids) do part, space, lgids
        space, lgids
    end
end

function Gridap.FESpaces.FEFunction(dV::DistributedFESpace, x)
    @abstractmethod
end

function Gridap.FESpaces.EvaluationFunction(dV::DistributedFESpace, x)
    @abstractmethod
end

"""
"""
function get_vector_type(a::DistributedFESpace)
  @abstractmethod
end

# Minimal FE interface
function Gridap.FESpaces.num_free_dofs(f::DistributedFESpace)
    f.gids.ngids
end

function Gridap.FESpaces.zero_free_values(f::DistributedFESpace)
    fv = Gridap.Algebra.allocate_vector(get_vector_type(f), f.gids)
    fill_entries!(fv, zero(eltype(fv)))
    fv
end

function Gridap.FESpaces.get_trial_fe_basis(f::DistributedFESpace)
    bases = DistributedData(f.spaces) do part, space
        get_trial_fe_basis(space)
    end
    DistributedCellBasis(bases)
end

# TO-DO: Better name?
struct DistributedFESpaceFromLocalFESpaces{V} <: DistributedFESpace
    vector_type::Type{V}
    spaces::DistributedData{<:FESpace}
    gids::DistributedIndexSet
end

function get_vector_type(a::DistributedFESpaceFromLocalFESpaces)
    a.vector_type
end

function Gridap.FESpaces.FEFunction(dV::DistributedFESpaceFromLocalFESpaces, x)
    dfree_vals = x[dV.gids]
    funs = DistributedData(dV.spaces, dfree_vals) do part, V, free_vals
        FEFunction(V, free_vals)
    end
    DistributedFEFunction(funs, x, dV)
end

function Gridap.FESpaces.EvaluationFunction(dV::DistributedFESpaceFromLocalFESpaces, x)
    dfree_vals = x[dV.gids]
    funs = DistributedData(dV.spaces, dfree_vals) do part, V, free_vals
        Gridap.FESpaces.EvaluationFunction(V, free_vals)
    end
    DistributedFEFunction(funs, x, dV)
end


#  Constructors
function Gridap.TrialFESpace(V::DistributedFESpaceFromLocalFESpaces, args...)
    spaces = DistributedData(V.spaces) do part, space
        TrialFESpace(space, args...)
    end
    DistributedFESpaceFromLocalFESpaces(get_vector_type(V), spaces, V.gids)
end

function DistributedFESpaceFromLocalFESpaces(::Type{V};
                                             model::DistributedDiscreteModel,
                                             reffe,
                                             kwargs...) where V
    function init_local_spaces(part, model)
        lspace = FESpace(model,reffe;kwargs...)
    end
    comm = get_comm(model)
    spaces = DistributedData(init_local_spaces, comm, model.models)
    DistributedFESpaceFromLocalFESpaces(V, model, spaces)
end

function DistributedFESpaceFromLocalFESpaces(::Type{V},
                                             model::DistributedDiscreteModel,
                                             spaces::DistributedData{<:FESpace}) where {V}
    gids = _compute_distributed_index_set(model, spaces)
    DistributedFESpaceFromLocalFESpaces(V, spaces, gids)
end

function _compute_distributed_index_set(
   model::DistributedDiscreteModel,
   spaces::DistributedData{<:FESpace})

    comm = get_comm(model)

    function init_lid_to_owner(part, lspace, cell_gids)
        nlids = num_free_dofs(lspace)
        lid_to_owner = zeros(Int, nlids)
        cell_to_part = cell_gids.lid_to_owner
        cell_to_lids = Table(get_cell_dof_ids(lspace))
        _fill_max_part_around!(lid_to_owner, cell_to_part, cell_to_lids)
        lid_to_owner
    end

    part_to_lid_to_owner = DistributedData{Vector{Int}}(init_lid_to_owner,
                                                        comm,
                                                        spaces,
                                                        model.gids)

    offsets, ngids = _compute_offsets_and_ngids(part_to_lid_to_owner)

    num_dofs_x_cell = compute_num_dofs_x_cell(comm,spaces)

    part_to_cell_to_owners = DistributedVector(init_cell_to_owners,
                                               model.gids, num_dofs_x_cell,
                                               spaces,
                                               part_to_lid_to_owner)

    exchange!(part_to_cell_to_owners)


    do_on_parts(update_lid_to_owner,
                part_to_lid_to_owner,
                spaces,
                part_to_cell_to_owners)

    part_to_lid_to_gid = _compute_part_to_lid_to_gid(model,
                                                     spaces,
                                                     num_dofs_x_cell,
                                                     part_to_lid_to_owner,
                                                     offsets)

    function init_free_gids(part, lid_to_gid, lid_to_owner, ngids)
        IndexSet(ngids, lid_to_gid, lid_to_owner)
    end

    gids = DistributedIndexSet(init_free_gids,
                               comm,
                               ngids,
                               part_to_lid_to_gid,
                               part_to_lid_to_owner,
                               ngids)
end

function _compute_offsets_and_ngids(part_to_lid_to_owner)
    comm = get_comm(part_to_lid_to_owner)
    function count_owned_lids(part, lid_to_owner)
        count(owner -> owner == part, lid_to_owner)
    end
    a = DistributedData{Int}(count_owned_lids, comm, part_to_lid_to_owner)
    part_to_num_oids = gather(a)
    if i_am_master(comm)
        ngids = sum(part_to_num_oids)
        _fill_offsets!(part_to_num_oids)
    else
        ngids = -1
    end
    offsets = scatter(comm, part_to_num_oids)
    part_to_ngids = scatter_value(comm, ngids)
    do_on_parts(comm, part_to_ngids) do part, lngids
        ngids = lngids
    end
    (offsets, ngids)
end

function compute_num_dofs_x_cell(comm, spaces)
  DistributedData(comm, spaces) do part, lspace
    cell_dofs = get_cell_dof_ids(lspace)
    [length(cell_dofs[i]) for i = 1:length(cell_dofs)]
  end
end

function _compute_part_to_lid_to_gid(model,
                                     spaces,
                                     num_dofs_x_cell,
                                     part_to_lid_to_owner,
                                     offsets)

    comm = get_comm(part_to_lid_to_owner)

    part_to_lid_to_gid = DistributedData{Vector{Int}}(
    init_lid_to_gids,comm,part_to_lid_to_owner,offsets)

    part_to_cell_to_gids = DistributedVector(init_cell_to_owners,
                                           model.gids,
                                           num_dofs_x_cell,
                                           spaces,
                                           part_to_lid_to_gid)

    exchange!(part_to_cell_to_gids)

    do_on_parts(
       update_lid_to_gid,
       part_to_lid_to_gid,
       part_to_lid_to_owner,
       spaces,
       part_to_cell_to_gids,
       model.gids)

    exchange!(part_to_cell_to_gids)

    do_on_parts(
       update_lid_to_owner,
       part_to_lid_to_gid,
       spaces,
       part_to_cell_to_gids)

    part_to_lid_to_gid
end

function init_lid_to_gids(part, lid_to_owner, offset)
    lid_to_gid = zeros(Int, length(lid_to_owner))
    _fill_owned_gids!(lid_to_gid, lid_to_owner, part, offset)
    lid_to_gid
end

function init_cell_to_owners(part,
  num_dofs_x_cell,
  lspace,
  lid_to_owner)
    ptrs = Vector{eltype(num_dofs_x_cell)}(undef,
              length(num_dofs_x_cell) + 1)
    ptrs[2:end] = num_dofs_x_cell[1:end]
    length_to_ptrs!(ptrs)
    data = Vector{eltype(lid_to_owner)}(undef, ptrs[end] - 1)

    cell_to_lids = get_cell_dof_ids(lspace)
    dlid_to_zero = zeros(eltype(lid_to_owner), num_dirichlet_dofs(lspace))
    cell_to_owners_from =
       lazy_map(Broadcasting(PosNegReindex(lid_to_owner,dlid_to_zero)),cell_to_lids)
    k = 1
    for i = 1:length(cell_to_owners_from)
        for j = 1:length(cell_to_owners_from[i])
            data[k] = cell_to_owners_from[i][j]
            k = k + 1
        end
    end
    return Table(data, ptrs)
end

function update_lid_to_gid(part, lid_to_gid, lid_to_owner, lspace, cell_to_gids, cell_gids)
    cell_to_lids = Table(get_cell_dof_ids(lspace))
    cell_to_owner = cell_gids.lid_to_owner
    _update_lid_to_gid!(
    lid_to_gid,cell_to_lids,cell_to_gids,cell_to_owner,lid_to_owner)
end

function _update_lid_to_gid!(lid_to_gid, cell_to_lids, cell_to_gids, cell_to_owner, lid_to_owner)
    for cell in 1:length(cell_to_lids)
        i_to_gid = cell_to_gids[cell]
        pini = cell_to_lids.ptrs[cell]
        pend = cell_to_lids.ptrs[cell + 1] - 1
        cellowner = cell_to_owner[cell]
        for (i, p) in enumerate(pini:pend)
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

function update_lid_to_owner(part, lid_to_owner, lspace, cell_to_owners)
    cell_to_lids = Table(get_cell_dof_ids(lspace))
    _update_lid_to_owner!(lid_to_owner, cell_to_lids, cell_to_owners)
end

function _update_lid_to_owner!(lid_to_owner, cell_to_lids, cell_to_owners)
    for cell in 1:length(cell_to_lids)
        i_to_owner = cell_to_owners[cell]
        pini = cell_to_lids.ptrs[cell]
        pend = cell_to_lids.ptrs[cell + 1] - 1
        for (i, p) in enumerate(pini:pend)
            lid = cell_to_lids.data[p]
            if lid > 0
                owner = i_to_owner[i]
                lid_to_owner[lid] = owner
            end
        end
    end
end

function _fill_owned_gids!(lid_to_gid, lid_to_owner, part, offset)
    o = offset
    for (lid, owner) in enumerate(lid_to_owner)
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

function _fill_max_part_around!(lid_to_owner, cell_to_owner, cell_to_lids)
    for cell in 1:length(cell_to_lids)
        cellowner = cell_to_owner[cell]
        pini = cell_to_lids.ptrs[cell]
        pend = cell_to_lids.ptrs[cell + 1] - 1
        for p in pini:pend
            lid = cell_to_lids.data[p]
            if lid > 0
                owner = lid_to_owner[lid]
                lid_to_owner[lid] = max(owner, cellowner)
            end
        end
    end
end

# FE Function
struct DistributedFEFunction
    funs::DistributedData
    vals::AbstractVector
    space::DistributedFESpace
end

# TO-DO Gridap.FESpaces.FEFunctionStyle(::Type{DistributedFEFunction}) = Val{true}()

get_distributed_data(u::DistributedFEFunction) = u.funs

Gridap.FESpaces.get_free_values(a::DistributedFEFunction) = a.vals

Gridap.FESpaces.get_fe_space(a::DistributedFEFunction) = a.space

# TO-DO Gridap.FESpaces.is_a_fe_function(a::DistributedFEFunction) = true

# Cell basis
struct DistributedCellBasis
    bases::DistributedData
end

# TO-DO Gridap.FESpaces.FECellBasisStyle(::Type{DistributedCellBasis}) = Val{true}()

get_distributed_data(u::DistributedCellBasis) = u.bases
