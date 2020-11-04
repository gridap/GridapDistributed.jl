struct MultiFieldDistributedFESpace{V} <: DistributedFESpace
    vector_type::Type{V}
    distributed_spaces::Vector{<:DistributedFESpace}
    spaces::DistributedData{<:MultiFieldFESpace}
    gids::DistributedIndexSet
end

function get_vector_type(a::MultiFieldDistributedFESpace)
    a.vector_type
end

function Gridap.MultiFieldFESpace(test_space::MultiFieldDistributedFESpace{V},
                                   trial_spaces::Vector{<:DistributedFESpace}) where V

    spaces = DistributedData(test_space.spaces, trial_spaces...) do part, lspace, spaces_and_gids...
        MultiFieldFESpace([s[1] for s in spaces_and_gids],MultiFieldStyle(lspace))
    end
    MultiFieldDistributedFESpace(V, trial_spaces, spaces, test_space.gids)
end


function Gridap.FESpaces.FEFunction(dV::MultiFieldDistributedFESpace{T}, x) where {T}
    _gen_multifield_distributed_fe_function(dV, x, FEFunction)
end


function _gen_multifield_distributed_fe_function(dV::MultiFieldDistributedFESpace{T}, x, f) where {T}
    single_fe_functions = DistributedFEFunction[]
    for (field, U) in enumerate(dV.distributed_spaces)
        free_values_i = restrict_to_field(dV, x, field)
        uhi = f(U, free_values_i)
        push!(single_fe_functions, uhi)
    end

    funs = DistributedData(get_comm(dV.distributed_spaces[1]),
                         dV.spaces, single_fe_functions...,) do part, V, fe_functions...
        mfv = zero_free_values(V)
        mf_lids = [i for i=1:length(mfv)]
        current = 1
        for (field_id,fun) in enumerate(fe_functions)
            fv = get_free_values(fun)
            sf_lids=Gridap.MultiField.restrict_to_field(V,mf_lids,field_id)
            for i = 1:length(sf_lids)
                mfv[sf_lids[i]] = fv[i]
                current = current + 1
            end
        end
        f(V, mfv)
    end
    multifield_fe_function = DistributedFEFunction(funs, x, dV)
    MultiFieldDistributedFEFunction(single_fe_functions,
                                  multifield_fe_function,
                                  dV)
end


function restrict_to_field(dV::MultiFieldDistributedFESpace, x::Vector, field)
    @assert isa(dV.gids, SequentialDistributedIndexSet)

    xi = Gridap.Algebra.allocate_vector(Vector{eltype(x)},
                                      dV.distributed_spaces[field].gids)

    do_on_parts(dV.spaces, dV.gids, xi, x, dV.distributed_spaces...) do part, mfspace, mfgids, xi, x, fspaces_and_gids...
        fspace = fspaces_and_gids[field][1]
        fgids  = fspaces_and_gids[field][2]
        mf_lids = [i for i=1:num_free_dofs(mfspace)]
        sf_lids = Gridap.MultiField.restrict_to_field(mfspace,mf_lids,field)
        for i = 1:num_free_dofs(fspace)
            if fgids.lid_to_owner[i] == part
                xi[fgids.lid_to_gid[i]] = x[mfgids.lid_to_gid[sf_lids[i]]]
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

    do_on_parts(dV.spaces, dV.gids, dV.distributed_spaces...) do part, mfspace, lmfgids, fspaces_and_gids...
        mf_lids = [i for i=1:num_free_dofs(mfspace)]
        sf_lids = Gridap.MultiField.restrict_to_field(mfspace,mf_lids,field)
        fspace = fspaces_and_gids[field][1]
        current=1
        for i = 1:num_free_dofs(fspace)
            if lmfgids.lid_to_owner[sf_lids[i]] == part
                mfis_gids[current]=mfgids.lid_to_gid_petsc[sf_lids[i]]-1
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
                                  distributed_spaces::Vector{<:DistributedFESpace}) where V

    spaces = DistributedData(distributed_spaces...) do part, spaces_and_gids...
        MultiFieldFESpace([s[1] for s in spaces_and_gids])
    end

    gids=_generate_multifield_gids(model,spaces,distributed_spaces)

    MultiFieldDistributedFESpace(get_vector_type(distributed_spaces[1]),
                               distributed_spaces,
                               spaces,
                               gids)
end


function Gridap.MultiFieldFESpace(model::DistributedDiscreteModel,
                                  distributed_spaces::Vector{<:DistributedFESpace},
                                  multifield_style::MultiFieldStyle) where V

    spaces = DistributedData(distributed_spaces...) do part, spaces_and_gids...
        MultiFieldFESpace([s[1] for s in spaces_and_gids],multifield_style)
    end

    gids=_generate_multifield_gids(model,spaces,distributed_spaces)

    MultiFieldDistributedFESpace(get_vector_type(distributed_spaces[1]),
                               distributed_spaces,
                               spaces,
                               gids)
end




function _generate_multifield_gids(model,spaces,distributed_spaces)
    function init_lid_to_owner(part, lspace, spaces_and_gids...)
        nlids = num_free_dofs(lspace)
        lid_to_owner = zeros(Int, nlids)
        mf_lids = [i for i=1:nlids]
        for (field_id,current_field_space_gids) in enumerate(spaces_and_gids)
            gids = current_field_space_gids[2]
            sf_lids=Gridap.MultiField.restrict_to_field(lspace,mf_lids,field_id)
            for i=1:length(sf_lids)
              lid_to_owner[sf_lids[i]]=gids.lid_to_owner[i]
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
end



# FE Function
struct MultiFieldDistributedFEFunction
    single_fe_functions::Vector{DistributedFEFunction}
    multifield_fe_function::DistributedFEFunction
    space::MultiFieldDistributedFESpace
end

Gridap.FESpaces.FEFunctionStyle(::Type{MultiFieldDistributedFEFunction}) = Val{true}()

get_distributed_data(u::MultiFieldDistributedFEFunction) =
     get_distributed_data(u.multifield_fe_function)

Gridap.FESpaces.get_free_values(a::MultiFieldDistributedFEFunction) =
     get_free_values(a.multifield_fe_function)

Gridap.FESpaces.get_fe_space(a::MultiFieldDistributedFEFunction) = a.space

Gridap.FESpaces.is_a_fe_function(a::MultiFieldDistributedFEFunction) = true
