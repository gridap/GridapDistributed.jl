struct MultiFieldDistributedFESpace{V} <: DistributedFESpace{V}
    vector_type::Type{V}
    distributed_spaces::Vector{<:DistributedFESpace}
    spaces::DistributedData{<:MultiFieldFESpace}
    gids::DistributedIndexSet
end

function Gridap.MultiFieldFESpace(test_space::MultiFieldDistributedFESpace{V},
                                   trial_spaces::Vector{<:DistributedFESpace{V}}) where V
    spaces = DistributedData(trial_spaces...) do part, spaces_and_gids...
        MultiFieldFESpace([s[1] for s in spaces_and_gids])
    end
    MultiFieldDistributedFESpace(V, trial_spaces, spaces, test_space.gids)
end


function Gridap.FESpaces.FEFunction(dV::MultiFieldDistributedFESpace{T}, x) where {T}
    _gen_multifield_distributed_fe_function(dV, x, FEFunction)
end


function _gen_multifield_distributed_fe_function(dV::MultiFieldDistributedFESpace{T}, x, f) where {T}
    single_fe_functions = DistributedFEFunction{T}[]
    for (field, U) in enumerate(dV.distributed_spaces)
        free_values_i = restrict_to_field(dV, x, field)
        uhi = f(U, free_values_i)
        push!(single_fe_functions, uhi)
    end

    funs = DistributedData(get_comm(dV.distributed_spaces[1]),
                         dV.spaces, single_fe_functions...,) do part, V, fe_functions...
        mfv = zero_free_values(V)
        current = 1
        for fun in fe_functions
            fv = get_free_values(fun)
            for i = 1:length(fv)
                mfv[current] = fv[i]
                current = current + 1
            end
        end
        f(V, mfv)
    end
    multifield_fe_function = DistributedFEFunction(funs, x, dV)
    MultiFieldDistributedFEFunction{T}(single_fe_functions,
                                  multifield_fe_function,
                                  dV)
end


function restrict_to_field(dV::MultiFieldDistributedFESpace, x::Vector, field)
    @assert isa(dV.gids, SequentialDistributedIndexSet)

    xi = Gridap.Algebra.allocate_vector(Vector{eltype(x)},
                                      dV.distributed_spaces[field].gids)

    do_on_parts(dV.spaces, dV.gids, xi, x, dV.distributed_spaces...) do part, mfspace, mfgids, xi, x, fspaces_and_gids...
        offset = 0
        for i = 1:field - 1
            fspace = fspaces_and_gids[i][1]
            offset = offset + num_free_dofs(fspace)
        end
        fspace = fspaces_and_gids[field][1]
        fgids  = fspaces_and_gids[field][2]
        for i = 1:num_free_dofs(fspace)
            if fgids.lid_to_owner[i] == part
                xi[fgids.lid_to_gid[i]] = x[mfgids.lid_to_gid[offset + i]]
            end
        end
    end
    xi
end

function restrict_to_field(dV::MultiFieldDistributedFESpace, x::GridapDistributedPETScWrappers.Vec{Float64}, field)
    fgids  = dV.distributed_spaces[field].gids
    mfgids = dV.gids
    @assert isa(fgids, MPIPETScDistributedIndexSet)

    xi = Gridap.Algebra.allocate_vector(GridapDistributedPETScWrappers.Vec{Float64},
                                      dV.distributed_spaces[field].gids)

    comm = get_comm(fgids)
    part = get_part(comm)

    fis_gids = [ GridapDistributedPETScWrappers.PetscInt(fgids.lid_to_gid_petsc[i] - 1)
                  for i = 1:length(fgids.lid_to_gid_petsc)
                     if fgids.parts.part.lid_to_owner[i] == part ]

    mfis_gids = Vector{GridapDistributedPETScWrappers.PetscInt}(undef,length(fis_gids))

    do_on_parts(dV.gids, dV.distributed_spaces...) do part, lmfgids, fspaces_and_gids...
        offset = 0
        for i = 1:field - 1
            fspace = fspaces_and_gids[i][1]
            offset = offset + num_free_dofs(fspace)
        end
        fspace = fspaces_and_gids[field][1]
        current=1
        for i = 1:num_free_dofs(fspace)
            if lmfgids.lid_to_owner[offset+i] == part
                mfis_gids[current]=mfgids.lid_to_gid_petsc[offset+i]-1
                current=current+1
            end
        end
    end

    fis  = GridapDistributedPETScWrappers.IS_(Float64, fis_gids; comm=comm.comm)
    mfis = GridapDistributedPETScWrappers.IS_(Float64, mfis_gids; comm=comm.comm)

    vscatter = GridapDistributedPETScWrappers.VecScatter(x, mfis, xi, fis)
    scatter!(vscatter,x,xi)
    xi
end



function Gridap.FESpaces.EvaluationFunction(dV::MultiFieldDistributedFESpace, x)
    _gen_multifield_distributed_fe_function(dV, x, EvaluationFEFunction)
end


function Gridap.MultiFieldFESpace(model::DistributedDiscreteModel,
                                  distributed_spaces::Vector{<:DistributedFESpace{V}}) where V

    spaces = DistributedData(distributed_spaces...) do part, spaces_and_gids...
        MultiFieldFESpace([s[1] for s in spaces_and_gids])
    end

    function init_lid_to_owner(part, lspace, spaces_and_gids...)
        nlids = num_free_dofs(lspace)
        lid_to_owner = zeros(Int, nlids)
        current_lid = 1
        for current_field_space_gids in spaces_and_gids
            gids = current_field_space_gids[2]
            for i = 1:length(gids.lid_to_owner)
                lid_to_owner[current_lid] = gids.lid_to_owner[i]
                current_lid += 1
            end
        end
        lid_to_owner
    end

    comm = get_comm(model)

    part_to_lid_to_owner = DistributedData{Vector{Int}}(init_lid_to_owner,
                                                      comm,
                                                      spaces,
                                                      distributed_spaces...)

    offsets, ngids = _compute_offsets_and_ngids(part_to_lid_to_owner)

    num_dofs_x_cell = compute_num_dofs_x_cell(comm, spaces)

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

    MultiFieldDistributedFESpace(V,
                               distributed_spaces,
                               spaces,
                               gids)
end

# FE Function
struct MultiFieldDistributedFEFunction{T}
    single_fe_functions::Vector{DistributedFEFunction{T}}
    multifield_fe_function::DistributedFEFunction{T}
    space::MultiFieldDistributedFESpace{T}
end

Gridap.FESpaces.FEFunctionStyle(::Type{MultiFieldDistributedFEFunction}) = Val{true}()

get_distributed_data(u::MultiFieldDistributedFEFunction) =
     get_distributed_data(u.multifield_fe_function)

Gridap.FESpaces.get_free_values(a::MultiFieldDistributedFEFunction) =
     get_free_values(a.multifield_fe_function)

Gridap.FESpaces.get_fe_space(a::MultiFieldDistributedFEFunction) = a.space

Gridap.FESpaces.is_a_fe_function(a::MultiFieldDistributedFEFunction) = true
