
struct DistributedMultiFieldFEFunction{A,B,C} <: DistributedGridapType
  field_fe_fun::A
  part_fe_fun::B
  free_values::C
  function DistributedMultiFieldFEFunction(
    field_fe_fun::AbstractVector{<:DistributedSingleFieldFEFunction},
    part_fe_fun::AbstractPData{<:MultiFieldFEFunction},
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
    part_fe_space::AbstractPData{<:MultiFieldFESpace},
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
  values = map_parts(f.part_fe_space,free_values.values) do u,x
    restrict_to_field(u,x,field)
  end
  gids = f.field_fe_space[field].gids
  PVector(values,gids)
end

function FESpaces.FEFunction(
  f::DistributedMultiFieldFESpace,x::AbstractVector,isconsistent=false)
  free_values = change_ghost(x,f.gids)
  # This will cause also the single-field components to be consistent
  local_vals = consistent_local_views(free_values,f.gids,isconsistent)
  part_fe_fun = map_parts(FEFunction,f.part_fe_space,local_vals)
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
  part_fe_fun = map_parts(EvaluationFunction,f.part_fe_space,local_vals)
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
  part_fe_fun = map_parts(local_vals,local_views(fe)) do x,f
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
  part_fe_fun = map_parts(local_vals,local_views(fe)) do x,f
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
  dirichlet_values::Vector{AbstractPData{<:AbstractVector}},
  fe::DistributedMultiFieldFESpace)
  local_vals = consistent_local_views(free_values,fe.gids,true)
  part_fe_fun = map_parts(local_vals,local_views(fe)) do x,f
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
  part_fe_fun = map_parts(local_spaces,local_objects...) do f,o...
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
struct DistributedMultiFieldFEBasis{A,B} <: DistributedGridapType
  field_fe_basis::A
  part_fe_basis::B
  function DistributedMultiFieldFEBasis(
    field_fe_basis::AbstractVector{<:DistributedCellField},
    part_fe_basis::AbstractPData{<:MultiFieldCellField})
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
  part_mbasis = map_parts(get_fe_basis,f.part_fe_space)
  field_fe_basis = DistributedCellField[]
  for i in 1:num_fields(f)
    basis_i = map_parts(b->b[i],part_mbasis)
    bi = DistributedCellField(basis_i)
    push!(field_fe_basis,bi)
  end
  DistributedMultiFieldFEBasis(field_fe_basis,part_mbasis)
end

function FESpaces.get_trial_fe_basis(f::DistributedMultiFieldFESpace)
  part_mbasis = map_parts(get_trial_fe_basis,f.part_fe_space)
  field_fe_basis = DistributedCellField[]
  for i in 1:num_fields(f)
    basis_i = map_parts(b->b[i],part_mbasis)
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
  p_f_space = map_parts(v,f_p_space...)
  p_mspace = map_parts(MultiFieldFESpace,p_f_space)
  gids = a.gids
  vector_type = a.vector_type
  DistributedMultiFieldFESpace(f_dspace,p_mspace,gids,vector_type)
end

# Factory

function MultiField.MultiFieldFESpace(
  f_dspace::Vector{<:DistributedSingleFieldFESpace})
  f_p_space = map(local_views,f_dspace)
  v(x...) = collect(x)
  p_f_space = map_parts(v,f_p_space...)
  p_mspace = map_parts(MultiFieldFESpace,p_f_space)
  gids = generate_multi_field_gids(f_dspace,p_mspace)
  vector_type = _find_vector_type(p_mspace,gids)
  DistributedMultiFieldFESpace(f_dspace,p_mspace,gids,vector_type)
end

function generate_multi_field_gids(
  f_dspace::Vector{<:DistributedSingleFieldFESpace},
  p_mspace::AbstractPData{<:MultiFieldFESpace})

  p_lids = map_parts(mspace->collect(get_free_dof_ids(mspace)),p_mspace)
  p_1lid_lid = map_parts(p_mspace,p_lids) do mspace, lids
    restrict_to_field(mspace,lids,1)
  end
  f_p_flid_lid = [p_1lid_lid]
  for f in 2:length(f_dspace)
    p_flid_lid = map_parts(p_mspace,p_lids) do mspace, lids
      restrict_to_field(mspace,lids,f)
    end
    push!(f_p_flid_lid,p_flid_lid)
  end
  f_frange = map(get_free_dof_ids,f_dspace)
  gids = generate_multi_field_gids(f_p_flid_lid,f_frange)
end

function generate_multi_field_gids(
  f_p_flid_lid::AbstractVector{<:AbstractPData{<:AbstractVector}},
  f_frange::AbstractVector{<:PRange})

  f_p_fiset = map(local_views,f_frange)

  v(x...) = collect(x)
  p_f_fiset = map_parts(v,f_p_fiset...)
  p_f_flid_lid = map_parts(v,f_p_flid_lid...)

  # Find the first gid of the multifield space in each part
  ngids = sum(map(num_gids,f_frange))
  p_noids = map_parts(f_fiset->sum(map(num_oids,f_fiset)),p_f_fiset)
  p_part = get_part_ids(p_noids)
  p_firstgid = xscan(+,p_noids,init=1)

  # Distributed gids to owned dofs
  p_lid_gid, p_lid_part = map_parts(
    p_f_flid_lid, p_f_fiset, p_firstgid) do f_flid_lid, f_fiset, firstgid
    nlids = sum(map(length,f_flid_lid))
    lid_gid = zeros(Int,nlids)
    lid_part = zeros(Int32,nlids)
    nf = length(f_fiset)
    gid = firstgid
    for f in 1:nf
      fiset = f_fiset[f]
      flid_lid = f_flid_lid[f]
      part = fiset.part
      for foid in 1:num_oids(fiset)
        flid = fiset.oid_to_lid[foid]
        lid = flid_lid[flid]
        lid_part[lid] = part
        lid_gid[lid] = gid
        gid += 1
      end
    end
    lid_gid,lid_part
  end

  # Now we need to propagate to ghost
  # to this end we use the already available
  # communicators in each of the single fields
  # We cannot use the cell wise dof like in the old version
  # since each field can be defined on an independent mesh.
  f_aux = map(frange->PVector{Int}(undef,frange),f_frange)
  propagate_to_ghost_multifield!(p_lid_gid,f_aux,f_p_flid_lid,f_p_fiset)
  propagate_to_ghost_multifield!(p_lid_part,f_aux,f_p_flid_lid,f_p_fiset)

  # Setup IndexSet
  p_iset = map_parts(IndexSet,p_part,p_lid_gid,p_lid_part)

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
  f_p_parts_snd = map(i->i.exchanger.parts_snd,f_frange)
  f_p_parts_rcv = map(i->i.exchanger.parts_rcv,f_frange)
  p_f_parts_snd = map_parts(v,f_p_parts_snd...)
  p_f_parts_rcv = map_parts(v,f_p_parts_rcv...)
  p_neigs_snd = map_parts(merge_neigs,p_f_parts_snd)
  p_neigs_rcv = map_parts(merge_neigs,p_f_parts_rcv)

  # Setup exchanger
  exchanger = Exchanger(p_iset,p_neigs_snd,p_neigs_rcv)

  # Setup the range
  ran = PRange(ngids,p_iset,exchanger)

  ran
end

function propagate_to_ghost_multifield!(
  p_lid_gid,f_gids,f_p_flid_lid,f_p_fiset)
  # Loop over fields
  nf = length(f_gids)
  for f in 1:nf
    # Write data into owned in single-field buffer
    gids = f_gids[f]
    p_flid_gid = gids.values
    p_flid_lid = f_p_flid_lid[f]
    p_fiset = f_p_fiset[f]
    map_parts(
      p_flid_gid,p_flid_lid,p_lid_gid,p_fiset) do flid_gid,flid_lid,lid_gid,fiset
      for foid in 1:num_oids(fiset)
        flid = fiset.oid_to_lid[foid]
        lid = flid_lid[flid]
        flid_gid[flid] = lid_gid[lid]
      end
    end
    # move to ghost
    exchange!(gids)
    # write again into multifield array on ghost ids
    map_parts(
      p_flid_gid,p_flid_lid,p_lid_gid,p_fiset) do flid_gid,flid_lid,lid_gid,fiset
      for fhid in 1:num_hids(fiset)
        flid = fiset.hid_to_lid[fhid]
        lid = flid_lid[flid]
        lid_gid[lid] = flid_gid[flid]
      end
    end
  end
end
