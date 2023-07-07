
struct DistributedMultiFieldFEFunction{A,B,C} <: GridapType
  field_fe_fun::A
  part_fe_fun::B
  free_values::C
  function DistributedMultiFieldFEFunction(
    field_fe_fun::AbstractVector{<:DistributedSingleFieldFEFunction},
    part_fe_fun::AbstractArray{<:MultiFieldFEFunction},
    free_values::AbstractVector)
    A = typeof(field_fe_fun)
    B = typeof(part_fe_fun)
    C = typeof(free_values)
    new{A,B,C}(field_fe_fun,part_fe_fun,free_values)
  end
end


function FESpaces.get_free_dof_values(uh::DistributedMultiFieldFEFunction)
  uh.free_values
end

local_views(a::DistributedMultiFieldFEFunction) = a.part_fe_fun
MultiField.num_fields(m::DistributedMultiFieldFEFunction) = length(m.field_fe_fun)
Base.iterate(m::DistributedMultiFieldFEFunction) = iterate(m.field_fe_fun)
Base.iterate(m::DistributedMultiFieldFEFunction,state) = iterate(m.field_fe_fun,state)
Base.getindex(m::DistributedMultiFieldFEFunction,field_id::Integer) = m.field_fe_fun[field_id]

local_views(a::Vector{<:DistributedCellField}) = [ai.fields for ai in a]

"""
"""
struct DistributedMultiFieldFESpace{A,B,C,D} <: DistributedFESpace
  field_fe_space::A
  part_fe_space::B
  gids::C
  vector_type::Type{D}
  function DistributedMultiFieldFESpace(
    field_fe_space::AbstractVector{<:DistributedSingleFieldFESpace},
    part_fe_space::AbstractArray{<:MultiFieldFESpace},
    gids::PRange,
    vector_type::Type{D}) where D
    A = typeof(field_fe_space)
    B = typeof(part_fe_space)
    C = typeof(gids)
    new{A,B,C,D}(field_fe_space,part_fe_space,gids,vector_type)
  end
end

local_views(a::DistributedMultiFieldFESpace) = a.part_fe_space
MultiField.num_fields(m::DistributedMultiFieldFESpace) = length(m.field_fe_space)
Base.iterate(m::DistributedMultiFieldFESpace) = iterate(m.field_fe_space)
Base.iterate(m::DistributedMultiFieldFESpace,state) = iterate(m.field_fe_space,state)
Base.getindex(m::DistributedMultiFieldFESpace,field_id::Integer) = m.field_fe_space[field_id]
Base.length(m::DistributedMultiFieldFESpace) = length(m.field_fe_space)

function FESpaces.get_vector_type(fs::DistributedMultiFieldFESpace)
  fs.vector_type
end

function FESpaces.get_free_dof_ids(fs::DistributedMultiFieldFESpace)
  fs.gids
end

function MultiField.restrict_to_field(
  f::DistributedMultiFieldFESpace,free_values::PVector,field::Integer)
  values = map(f.part_fe_space,partition(free_values)) do u,x
    restrict_to_field(u,x,field)
  end
  gids = f.field_fe_space[field].gids
  PVector(values,partition(gids))
end

function FESpaces.FEFunction(
  f::DistributedMultiFieldFESpace,x::AbstractVector,isconsistent=false)
  free_values = change_ghost(x,f.gids)
  # This will cause also the single-field components to be consistent
  local_vals = consistent_local_views(free_values,f.gids,isconsistent)
  part_fe_fun = map(FEFunction,f.part_fe_space,local_vals)
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(f)
    free_values_i = restrict_to_field(f,free_values,i)
    fe_space_i = f.field_fe_space[i]
    fe_fun_i = FEFunction(fe_space_i,free_values_i,true)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.EvaluationFunction(
  f::DistributedMultiFieldFESpace,x::AbstractVector,isconsistent=false)
  free_values = change_ghost(x,f.gids)
  # This will cause also the single-field components to be consistent
  local_vals = consistent_local_views(free_values,f.gids,false)
  part_fe_fun = map(EvaluationFunction,f.part_fe_space,local_vals)
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(f)
    free_values_i = restrict_to_field(f,free_values,i)
    fe_space_i = f.field_fe_space[i]
    fe_fun_i = EvaluationFunction(fe_space_i,free_values_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate(objects,fe::DistributedMultiFieldFESpace)
  free_values = zero_free_values(fe)
  interpolate!(objects,free_values,fe)
end

function FESpaces.interpolate!(objects,free_values::AbstractVector,fe::DistributedMultiFieldFESpace)
  local_vals = consistent_local_views(free_values,fe.gids,true)
  part_fe_fun = map(local_vals,local_views(fe)) do x,f
    interpolate!(objects,x,f)
  end
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(fe)
    free_values_i = restrict_to_field(fe,free_values,i)
    fe_space_i = fe.field_fe_space[i]
    fe_fun_i = FEFunction(fe_space_i,free_values_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate_everywhere(objects,fe::DistributedMultiFieldFESpace)
  free_values = zero_free_values(fe)
  local_vals = consistent_local_views(free_values,fe.gids,true)
  part_fe_fun = map(local_vals,local_views(fe)) do x,f
    interpolate!(objects,x,f)
  end
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(fe)
    free_values_i = restrict_to_field(fe,free_values,i)
    fe_space_i = fe.field_fe_space[i]
    dirichlet_values_i = zero_dirichlet_values(fe_space_i)
    fe_fun_i = interpolate_everywhere!(objects[i], free_values_i,dirichlet_values_i,fe_space_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate_everywhere!(
  objects,free_values::AbstractVector,
  dirichlet_values::Vector{AbstractArray{<:AbstractVector}},
  fe::DistributedMultiFieldFESpace)
  local_vals = consistent_local_views(free_values,fe.gids,true)
  part_fe_fun = map(local_vals,local_views(fe)) do x,f
    interpolate!(objects,x,f)
  end
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(fe)
    free_values_i = restrict_to_field(fe,free_values,i)
    dirichlet_values_i = dirichlet_values[i]
    fe_space_i = fe.field_fe_space[i]
    fe_fun_i = interpolate_everywhere!(objects[i], free_values_i,dirichlet_values_i,fe_space_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate_everywhere(
  objects::Vector{<:DistributedCellField},fe::DistributedMultiFieldFESpace)
  local_objects = local_views(objects)
  local_spaces = local_views(fe)
  part_fe_fun = map(local_spaces,local_objects...) do f,o...
    interpolate_everywhere(o,f)
  end
  free_values = zero_free_values(fe)
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(fe)
    free_values_i = restrict_to_field(fe,free_values,i)
    fe_space_i = fe.field_fe_space[i]
    dirichlet_values_i = get_dirichlet_dof_values(fe_space_i)
    fe_fun_i = interpolate_everywhere!(objects[i], free_values_i,dirichlet_values_i,fe_space_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end


"""
"""
struct DistributedMultiFieldFEBasis{A,B} <: GridapType
  field_fe_basis::A
  part_fe_basis::B
  function DistributedMultiFieldFEBasis(
    field_fe_basis::AbstractVector{<:DistributedCellField},
    part_fe_basis::AbstractArray{<:MultiFieldCellField})
    A = typeof(field_fe_basis)
    B = typeof(part_fe_basis)
    new{A,B}(field_fe_basis,part_fe_basis)
  end
end

local_views(a::DistributedMultiFieldFEBasis) = a.part_fe_basis
MultiField.num_fields(m::DistributedMultiFieldFEBasis) = length(m.field_fe_basis)
Base.iterate(m::DistributedMultiFieldFEBasis) = iterate(m.field_fe_basis)
Base.iterate(m::DistributedMultiFieldFEBasis,state) = iterate(m.field_fe_basis,state)
Base.getindex(m::DistributedMultiFieldFEBasis,field_id::Integer) = m.field_fe_basis[field_id]

function FESpaces.get_fe_basis(f::DistributedMultiFieldFESpace)
  part_mbasis = map(get_fe_basis,f.part_fe_space)
  field_fe_basis = DistributedCellField[]
  for i in 1:num_fields(f)
    basis_i = map(b->b[i],part_mbasis)
    bi = DistributedCellField(basis_i)
    push!(field_fe_basis,bi)
  end
  DistributedMultiFieldFEBasis(field_fe_basis,part_mbasis)
end

function FESpaces.get_trial_fe_basis(f::DistributedMultiFieldFESpace)
  part_mbasis = map(get_trial_fe_basis,f.part_fe_space)
  field_fe_basis = DistributedCellField[]
  for i in 1:num_fields(f)
    basis_i = map(b->b[i],part_mbasis)
    bi = DistributedCellField(basis_i)
    push!(field_fe_basis,bi)
  end
  DistributedMultiFieldFEBasis(field_fe_basis,part_mbasis)
end

function FESpaces.TrialFESpace(objects,a::DistributedMultiFieldFESpace)
  TrialFESpace(a,objects)
end

function FESpaces.TrialFESpace(a::DistributedMultiFieldFESpace,objects)
  f_dspace_test = a.field_fe_space
  f_dspace = map( arg -> TrialFESpace(arg[1],arg[2]), zip(f_dspace_test,objects) )
  f_p_space = map(local_views,f_dspace)
  v(x...) = collect(x)
  p_f_space = map(v,f_p_space...)
  p_mspace = map(MultiFieldFESpace,p_f_space)
  gids = a.gids
  vector_type = a.vector_type
  DistributedMultiFieldFESpace(f_dspace,p_mspace,gids,vector_type)
end

# Factory

function MultiField.MultiFieldFESpace(
  f_dspace::Vector{<:DistributedSingleFieldFESpace})
  f_p_space = map(local_views,f_dspace)
  v(x...) = collect(x)
  p_f_space = map(v,f_p_space...)
  p_mspace = map(MultiFieldFESpace,p_f_space)
  gids = generate_multi_field_gids(f_dspace,p_mspace)
  vector_type = _find_vector_type(p_mspace,gids)
  DistributedMultiFieldFESpace(f_dspace,p_mspace,gids,vector_type)
end

function generate_multi_field_gids(
  f_dspace::Vector{<:DistributedSingleFieldFESpace},
  p_mspace::AbstractArray{<:MultiFieldFESpace})

  p_lids = map(mspace->collect(get_free_dof_ids(mspace)),p_mspace)
  p_1lid_lid = map(p_mspace,p_lids) do mspace, lids
    restrict_to_field(mspace,lids,1)
  end
  f_p_flid_lid = [p_1lid_lid]
  for f in 2:length(f_dspace)
    p_flid_lid = map(p_mspace,p_lids) do mspace, lids
      restrict_to_field(mspace,lids,f)
    end
    push!(f_p_flid_lid,p_flid_lid)
  end
  f_frange = map(get_free_dof_ids,f_dspace)
  gids = generate_multi_field_gids(f_p_flid_lid,f_frange)
end

function generate_multi_field_gids(
  f_p_flid_lid::AbstractVector{<:AbstractArray{<:AbstractVector}},
  f_frange::AbstractVector{<:PRange})

  f_p_fiset = map(local_views,f_frange)

  v(x...) = collect(x)
  p_f_fiset = map(v,f_p_fiset...)
  p_f_flid_lid = map(v,f_p_flid_lid...)

  # Find the first gid of the multifield space in each part
  ngids = sum(map(length,f_frange))
  p_noids = map(f_fiset->sum(map(own_length,f_fiset)),p_f_fiset)
  p_firstgid = scan(+,p_noids,type=:exclusive,init=one(eltype(p_noids)))

  # Distributed gids to owned dofs
  p_lid_gid, p_lid_part = map(
    p_f_flid_lid, p_f_fiset, p_firstgid) do f_flid_lid, f_fiset, firstgid
    nlids = sum(map(length,f_flid_lid))
    lid_gid = zeros(Int,nlids)
    lid_part = zeros(Int32,nlids)
    nf = length(f_fiset)
    gid = firstgid
    for f in 1:nf
      fiset = f_fiset[f]
      fiset_owner_to_local = own_to_local(fiset)
      flid_lid = f_flid_lid[f]
      part = part_id(fiset)
      for foid in 1:own_length(fiset)
        flid = fiset_owner_to_local[foid]
        lid = flid_lid[flid]
        lid_part[lid] = part
        lid_gid[lid] = gid
        gid += 1
      end
    end
    lid_gid,lid_part
  end |> tuple_of_arrays

  # Now we need to propagate to ghost
  # to this end we use the already available
  # communicators in each of the single fields
  # We cannot use the cell wise dof like in the old version
  # since each field can be defined on an independent mesh.
  f_aux_gids = map(frange->PVector{Vector{eltype(eltype(p_lid_gid))}}(undef,partition(frange)),f_frange)
  f_aux_part = map(frange->PVector{Vector{eltype(eltype(p_lid_part))}}(undef,partition(frange)),f_frange)
  propagate_to_ghost_multifield!(p_lid_gid,f_aux_gids,f_p_flid_lid,f_p_fiset)
  propagate_to_ghost_multifield!(p_lid_part,f_aux_part,f_p_flid_lid,f_p_fiset)

  p_iset = map(partition(f_frange[1]),p_lid_gid,p_lid_part) do indices,
                                                               lid_to_gid,
                                                               lid_to_owner
     me = part_id(indices)
     LocalIndices(ngids,me,lid_to_gid,lid_to_owner)
  end

  # Merge neighbors
  function merge_neigs(f_neigs)
    dict = Dict{Int32,Int32}()
    for f in 1:length(f_neigs)
      for neig in f_neigs[f]
        dict[neig] = neig
      end
    end
    collect(keys(dict))
  end
  
  f_p_parts_snd, f_p_parts_rcv = map(x->assembly_neighbors(partition(x)),f_frange) |> tuple_of_arrays
  p_f_parts_snd = map(v,f_p_parts_snd...)
  p_f_parts_rcv = map(v,f_p_parts_rcv...)
  p_neigs_snd = map(merge_neigs,p_f_parts_snd)
  p_neigs_rcv = map(merge_neigs,p_f_parts_rcv)
  
  exchange_graph = ExchangeGraph(p_neigs_snd,p_neigs_rcv)
  assembly_neighbors(p_iset;neighbors=exchange_graph)
  
  PRange(p_iset)
end

function propagate_to_ghost_multifield!(
  p_lid_gid,f_gids,f_p_flid_lid,f_p_fiset)
  # Loop over fields
  nf = length(f_gids)
  for f in 1:nf
    # Write data into owned in single-field buffer
    gids = f_gids[f]
    p_flid_gid = gids.vector_partition
    p_flid_lid = f_p_flid_lid[f]
    p_fiset = f_p_fiset[f]
    map(
      p_flid_gid,p_flid_lid,p_lid_gid,p_fiset) do flid_gid,flid_lid,lid_gid,fiset
      fiset_own_to_local = own_to_local(fiset)
      for foid in 1:own_length(fiset)
        flid = fiset_own_to_local[foid]
        lid = flid_lid[flid]
        flid_gid[flid] = lid_gid[lid]
      end
    end
    # move to ghost
    cache=fetch_vector_ghost_values_cache(partition(gids),p_fiset)
    fetch_vector_ghost_values!(partition(gids),cache) |> wait
    # write again into multifield array on ghost ids
    map(
      p_flid_gid,p_flid_lid,p_lid_gid,p_fiset) do flid_gid,flid_lid,lid_gid,fiset
      fiset_ghost_to_local=ghost_to_local(fiset)
      for fhid in 1:ghost_length(fiset)
        flid = fiset_ghost_to_local[fhid]
        lid = flid_lid[flid]
        lid_gid[lid] = flid_gid[flid]
      end
    end
  end
end
