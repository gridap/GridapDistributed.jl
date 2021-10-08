
struct DistributedMultiFieldFEFunction{A,B,C} <: GridapType
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

struct DistributedMultiFieldFESpace{A,B,C} <: DistributedFESpace
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

function FESpaces.MultiFieldStyle(f::DistributedMultiFieldFESpace)
  MultiFieldStyle(get_part(f.part_fe_space))
end

function FESpaces.get_vector_type(fs::DistributedMultiFieldFESpace)
  fs.vector_type
end

function FESpaces.get_free_dof_ids(fs::DistributedMultiFieldFESpace)
  fs.gids
end

function FESpaces.restrict_to_field(
  f::DistributedMultiFieldFESpace,free_values::PVector,field::Integer)
  values = map_parts(f.part_fe_space,free_values.values) do u,x
    restrict_to_field(u,x,field)
  end
  gids = f.field_fe_space[field].gids
  PVector(values,gids)
end

function FESpaces.FEFunction(
  f::DistributedSingleFieldFESpace,free_values::AbstractVector)
  local_vals = consistent_local_views(free_values,f.gids)
  part_fe_fun = map_parts(FEFunction,f.part_fe_space,local_vals)
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(f)
    free_values_i = restrict_to_field(f,free_values,i)
    fe_space_i = f.field_fe_space[i]
    fe_fun_i = FEFunction(fe_space_i,free_values_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.EvaluationFunction(
  f::DistributedSingleFieldFESpace,free_values::AbstractVector)
  local_vals = consistent_local_views(free_values,f.gids)
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

struct DistributedMultiFieldFEBasis{A,B} <: GridapType
  field_fe_basis::A
  part_fe_basis::B
  function DistributedMultiFieldFEBasis(
    field_fe_basis::AbstractVector{<:DistributedCellField},
    part_fe_basis::AbstractPData{<:MultiFieldFEBasis})
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
  part_fe_basis = map_parts(get_fe_basis,f.part_fe_space)
  field_fe_basis = DistributedCellField[]
  for i in 1:num_fields(f)
    basis_i = get_fe_basis(f.field_fe_space[i])
    push!(field_fe_basis,basis_i)
  end
  DistributedMultiFieldFEBasis(field_fe_basis,part_fe_basis)
end

function FESpaces.get_trial_fe_basis(f::DistributedMultiFieldFESpace)
  part_fe_basis = map_parts(get_trial_fe_basis,f.part_fe_space)
  field_fe_basis = DistributedCellField[]
  for i in 1:num_fields(f)
    basis_i = get_trial_fe_basis(f.field_fe_space[i])
    push!(field_fe_basis,basis_i)
  end
  DistributedMultiFieldFEBasis(field_fe_basis,part_fe_basis)
end

function FESpaces.interpolate(u,f::DistributedMultiFieldFESpace)
  @notimplemented
end

# Factory

function MultiField.MultiFieldFESpace(
  spaces::Vector{<:DistributedSingleFieldFESpace})


end

function generate_multi_field_gids(
  p_f_flid_lid::AbstractPData{<:AbstractVector{<:AbstractVector}}
  f_frange::AbstractVector{<:PRange})

  f_p_fiset = map(local_views,f_frange)

  v(x...) = collect(x)
  p_f_fiset = map_parts(v,f_p_fiset...)
  f_p_flid_lid = collect(map_parts(identity,p_f_flid_lid))

  # Find the first gid of the multifield space in each part
  ngids = sum(map(num_gids,f_frange))
  p_noids = map_parts(f_fiset->sum(map(num_oids,f_fiset)),p_f_fiset)
  p_part = get_part_ids(p_noids)
  p_firstgid = xscan_all(+,p_noids,init=1)

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
      for foid in 1:num_lids(fiset)
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
    for f in length(f_neigs)
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
  ran = PRange(ngdofs,p_iset,exchanger)

  ran
end

function propagate_to_ghost_multifield!(
  p_lid_gid,f_gids,f_p_flid_lid,f_p_fiset)
  # Loop over fields
  nf = length(f_gids)
  for f in 1:nf
    # Write data into owned in single-field buffer
    gids = f_gids[f]
    p_flid_gid = local_view(gids)
    p_flid_lid = f_p_flid_lid[f]
    p_fiset = f_p_fiset[f]
    map_parts(
      p_flid_gid,p_flid_lid,p_lid_gid,p_fiset) do flid_gid,flid_lid,lid_gid,fiset
      for foid in 1:num_oids(fiset)
        flid = fiset.oid_to_lid[foid]
        lid = flid_lid[flid]
        flid_gid[foid] = lid_gid[lid]
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
        lid_gid[lid] = flid_gid[foid]
      end
    end
  end
end

function generate_multi_field_gids(
  f_frange::AbstractVector{<:PRange})

  f_gids = map(i->PVector{Int}(undef,i),f_frange)
  f_p_gids = map(local_views,f_gids)
  f_p_fids = map(local_views,f_frange)
  f_p_parts_snd = map(i->i.exchanger.parts_snd,f_frange)
  f_p_parts_rcv = map(i->i.exchanger.parts_rcv,f_frange)

  v(x...) = collect(x)
  p_f_fids = map_parts(v,f_p_fids...)
  p_f_gids = map_parts(v,f_p_gids...)
  p_f_parts_snd = map_parts(v,f_p_parts_snd...)
  p_f_parts_rcv = map_parts(v,f_p_parts_rcv...)

  ngids = sum(map(num_gids,f_franges))
  p_noids = map_parts(f_fids->sum(map(num_oids,f_fids)),p_f_fids)
  p_part = get_part_ids(p_noids)
  p_firstgid = xscan_all(+,p_noids,init=1)

  p_f_firstgid = map_parts(p_firstgid,p_f_fids) do firstgid, f_fids
    c = field_gids
    nf = length(f_fids)
    f_firstgid = zeros(Int,nf)
    for f in 1:nf
      f_firstgid[f] = c
      c += num_oids(f_fids[f])
    end
    f_firstgid
  end

  for f in 1:length(f_frange)
    p_gids = f_p_gids[f]
    p_fids = f_p_fids[f]
    map_parts(p_gids,p_fids,p_f_firstgid) do gids,fids,f_firstgid
      c = f_firstgid[f]
      for foid in 1:num_oids(fids)
        flid = fids.oid_to_lid[foid]
        gids[flid] = c
        c += 1
      end
    end
    exchange!(f_gids[f])
  end

  p_hid_gid, p_hid_part = map_parts(p_f_fids,p_f_gids) do f_fids,f_gids
    nf = length(f_fids)
    nhids = sum(map(num_hids,f_ids))
    hid_gid = zeros(Int,nhids)
    hid_part = zeros(Int32,nhids)
    c = 1
    for f in 1:nf
      fids = f_fids[f]
      gids = f_gids[f]
      for fhid in 1:num_hids(fids)
        flid = fids.hid_to_lid[fhid]
        gid = gids[flid]
        part = fids.lid_to_part[flid]
        hid_to_gid[c] = gid
        hid_to_part[c] = part
        c += 1
      end
    end
    hid_gid, hid_part
  end

  function merge_neigs(f_neigs)
    dict = Dict{Int32,Int32}()
    for f in length(f_neigs)
      for neig in f_neigs[f]
        dict[neig] = neig
      end
    end
    collect(keys(dict))
  end
  p_neigs_snd = map_parts(merge_neigs,p_f_parts_snd)
  p_neigs_rcv = map_parts(merge_neigs,p_f_parts_rcv)

  gids = PRange(
    p_part,
    ngids,
    p_noids,
    p_firstgid,
    p_hid_gid,
    p_hid_part,
    p_neigs_snd,
    p_neigs_rcv)

end

