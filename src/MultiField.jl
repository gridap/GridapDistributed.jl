
# DistributedMultiFieldCellField
struct DistributedMultiFieldCellField{A,B,C} <: CellField
  field_fe_fun::A
  part_fe_fun::B
  metadata::C
  function DistributedMultiFieldCellField(
    field_fe_fun::AbstractVector{<:DistributedCellField},
    part_fe_fun::AbstractArray{<:CellField},
    metadata=nothing
  )
    A = typeof(field_fe_fun)
    B = typeof(part_fe_fun)
    C = typeof(metadata)
    new{A,B,C}(field_fe_fun,part_fe_fun,metadata)
  end
end

function CellData.get_triangulation(a::DistributedMultiFieldCellField)
  trians = map(get_triangulation,a.field_fe_fun)
  @check all(map(t -> t === first(trians), trians))
  return first(trians)
end

function CellData.DomainStyle(::Type{<:DistributedMultiFieldCellField{A,B}}) where {A,B}
  DomainStyle(eltype(B))
end

local_views(a::DistributedMultiFieldCellField) = a.part_fe_fun
local_views(a::Vector{<:DistributedCellField}) = map(local_views,a)

MultiField.num_fields(m::DistributedMultiFieldCellField) = length(m.field_fe_fun)
Base.iterate(m::DistributedMultiFieldCellField) = iterate(m.field_fe_fun)
Base.iterate(m::DistributedMultiFieldCellField,state) = iterate(m.field_fe_fun,state)
Base.getindex(m::DistributedMultiFieldCellField,field_id::Integer) = m.field_fe_fun[field_id]

function LinearAlgebra.dot(a::DistributedMultiFieldCellField,b::DistributedMultiFieldCellField)
  @check num_fields(a) == num_fields(b)
  return sum(map(dot,a.field_fe_fun,b.field_fe_fun))
end

# DistributedMultiFieldFEFunction

const DistributedMultiFieldFEFunction{A,B,T} = DistributedMultiFieldCellField{A,B,DistributedFEFunctionData{T}}

function DistributedMultiFieldFEFunction(
  field_fe_fun::AbstractVector{<:DistributedSingleFieldFEFunction},
  part_fe_fun::AbstractArray{<:MultiFieldFEFunction},
  free_values::AbstractVector
)
  metadata = DistributedFEFunctionData(free_values)
  DistributedMultiFieldCellField(field_fe_fun,part_fe_fun,metadata)
end

function FESpaces.get_free_dof_values(uh::DistributedMultiFieldFEFunction)
  uh.metadata.free_values
end

# DistributedMultiFieldFESpace

"""
"""
struct DistributedMultiFieldFESpace{MS,A,B,C,D} <: DistributedFESpace
  multi_field_style::MS
  field_fe_space::A
  part_fe_space::B
  gids::C
  vector_type::Type{D}
  function DistributedMultiFieldFESpace(
    field_fe_space::AbstractVector{<:DistributedSingleFieldFESpace},
    part_fe_space::AbstractArray{<:MultiFieldFESpace{MS}},
    gids::Union{<:PRange,<:BlockPRange},
    vector_type::Type{D}) where {D,MS}
    A = typeof(field_fe_space)
    B = typeof(part_fe_space)
    C = typeof(gids)
    new{MS,A,B,C,D}(MS(),field_fe_space,part_fe_space,gids,vector_type)
  end
end

function CellData.get_triangulation(a::DistributedMultiFieldFESpace)
  trians = map(get_triangulation,a.field_fe_space)
  @check all(map(t -> t === first(trians), trians))
  return first(trians)
end

MultiField.MultiFieldStyle(::Type{<:DistributedMultiFieldFESpace{MS}}) where MS = MS()
MultiField.MultiFieldStyle(a::DistributedMultiFieldFESpace) = MultiField.MultiFieldStyle(typeof(a))

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
  f::DistributedMultiFieldFESpace,free_values::AbstractVector,field::Integer
)
  values = map(local_views(f),partition(free_values)) do u,fv
    restrict_to_field(u,fv,field)
  end
  gids = get_free_dof_ids(f[field])
  PVector(values,partition(gids))
end

function FESpaces.zero_dirichlet_values(f::DistributedMultiFieldFESpace)
  map(zero_dirichlet_values,f.field_fe_space)
end

function FESpaces.get_dirichlet_dof_values(f::DistributedMultiFieldFESpace)
  return map(get_dirichlet_dof_values,f.field_fe_space)
end

function FESpaces.FEFunction(
  f::DistributedMultiFieldFESpace,free_values::AbstractVector,isconsistent=false
)
  dirichlet_values = get_dirichlet_dof_values(f)
  return FEFunction(f,free_values,dirichlet_values,isconsistent)
end

function FESpaces.FEFunction(
  f::DistributedMultiFieldFESpace,
  _free_values::AbstractVector,
  dirichlet_values::AbstractArray{<:AbstractVector},
  isconsistent=false
)
  free_values = change_ghost(_free_values,f.gids;is_consistent=isconsistent,make_consistent=true)

  # Create distributed single field functions
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(f)
    free_values_i = restrict_to_field(f,free_values,i)
    dirichlet_values_i = dirichlet_values[i]
    fe_space_i = f.field_fe_space[i]
    fe_fun_i = FEFunction(fe_space_i,free_values_i,dirichlet_values_i,true)
    push!(field_fe_fun,fe_fun_i)
  end

  # Retrieve the local multifield views
  part_sf_fe_funs = map(local_views,field_fe_fun)
  part_fe_fun = map(local_views(f),partition(free_values),part_sf_fe_funs...) do space,fv,part_sf_fe_funs...
    MultiFieldFEFunction(fv,space,[part_sf_fe_funs...])
  end

  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.EvaluationFunction(
  f::DistributedMultiFieldFESpace,
  _free_values::AbstractVector,
  isconsistent=false
)
  free_values = change_ghost(_free_values,f.gids;is_consistent=isconsistent,make_consistent=true)
  
  # Create distributed single field functions
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(f)
    free_values_i = restrict_to_field(f,free_values,i)
    fe_space_i = f.field_fe_space[i]
    fe_fun_i = EvaluationFunction(fe_space_i,free_values_i)
    push!(field_fe_fun,fe_fun_i)
  end

  # Retrieve the local multifield views
  part_sf_fe_funs = map(local_views,field_fe_fun)
  part_fe_fun = map(local_views(f),partition(free_values),part_sf_fe_funs...) do space,fv,part_sf_fe_funs...
    MultiFieldFEFunction(fv,space,[part_sf_fe_funs...])
  end

  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate(objects,space::DistributedMultiFieldFESpace)
  free_values = zero_free_values(space)
  interpolate!(objects,free_values,space)
end

function FESpaces.interpolate!(objects,free_values::AbstractVector,space::DistributedMultiFieldFESpace)
  msg = "free_values and FESpace have incompatible index partitions."
  @check PartitionedArrays.matching_local_indices(axes(free_values,1),get_free_dof_ids(space)) msg

  # Interpolate each field
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(space)
    free_values_i = restrict_to_field(space,free_values,i)
    fe_space_i = space.field_fe_space[i]
    fe_fun_i = interpolate!(objects[i], free_values_i, fe_space_i)
    push!(field_fe_fun,fe_fun_i)
  end

  # Retrieve the local multifield views
  part_sf_fe_funs = map(local_views,field_fe_fun)
  part_fe_fun = map(local_views(space),partition(free_values),part_sf_fe_funs...) do space,fv,part_sf_fe_funs...
    MultiFieldFEFunction(fv,space,[part_sf_fe_funs...])
  end

  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate_everywhere(objects,fe::DistributedMultiFieldFESpace)
  free_values = zero_free_values(fe)
  dirichlet_values = zero_dirichlet_values(fe)
  return interpolate_everywhere!(objects,free_values,dirichlet_values,fe)
end

function FESpaces.interpolate_everywhere!(
  objects,
  free_values::AbstractVector,
  dirichlet_values::Vector{<:AbstractArray{<:AbstractVector}},
  space::DistributedMultiFieldFESpace
)
  msg = "free_values and FESpace have incompatible index partitions."
  @check PartitionedArrays.matching_local_indices(axes(free_values,1),get_free_dof_ids(space)) msg
  
  # Interpolate each field
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(space)
    free_values_i = restrict_to_field(space,free_values,i)
    dirichlet_values_i = dirichlet_values[i]
    fe_space_i = space.field_fe_space[i]
    fe_fun_i = interpolate_everywhere!(objects[i], free_values_i, dirichlet_values_i,fe_space_i)
    push!(field_fe_fun,fe_fun_i)
  end

  # Retrieve the local multifield views
  part_sf_fe_funs = map(local_views,field_fe_fun)
  part_fe_fun = map(local_views(space),partition(free_values),part_sf_fe_funs...) do space,fv,part_sf_fe_funs...
    MultiFieldFEFunction(fv,space,[part_sf_fe_funs...])
  end

  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.TrialFESpace(objects,a::DistributedMultiFieldFESpace)
  TrialFESpace(a,objects)
end

function FESpaces.TrialFESpace(a::DistributedMultiFieldFESpace{MS},objects) where MS
  f_dspace_test = a.field_fe_space
  f_dspace = map( arg -> TrialFESpace(arg[1],arg[2]), zip(f_dspace_test,objects) )
  f_p_space = map(local_views,f_dspace)
  v(x...) = collect(x)
  p_f_space = map(v,f_p_space...)
  p_mspace  = map(s->MultiFieldFESpace(s;style=MS()),p_f_space)
  gids = a.gids
  vector_type = a.vector_type
  DistributedMultiFieldFESpace(f_dspace,p_mspace,gids,vector_type)
end

# DistributedMultiFieldFEBasis

const DistributedMultiFieldFEBasis{A} = DistributedMultiFieldCellField{A,<:AbstractArray{<:FEBasis}}

function FESpaces.get_fe_basis(f::DistributedMultiFieldFESpace)
  part_mbasis = map(get_fe_basis,f.part_fe_space)
  field_fe_basis = map(1:num_fields(f)) do i
    space_i = f.field_fe_space[i]
    basis_i = map(b->b[i],part_mbasis)
    DistributedCellField(basis_i,get_triangulation(space_i))
  end
  DistributedMultiFieldCellField(field_fe_basis,part_mbasis)
end

function FESpaces.get_trial_fe_basis(f::DistributedMultiFieldFESpace)
  part_mbasis = map(get_trial_fe_basis,f.part_fe_space)
  field_fe_basis = map(1:num_fields(f)) do i
    space_i = f.field_fe_space[i]
    basis_i = map(b->b[i],part_mbasis)
    DistributedCellField(basis_i,get_triangulation(space_i))
  end
  DistributedMultiFieldCellField(field_fe_basis,part_mbasis)
end

# Factory

function MultiField.MultiFieldFESpace(
  f_dspace::Vector{<:DistributedSingleFieldFESpace};split_own_and_ghost=false, kwargs...)
  f_p_space = map(local_views,f_dspace)
  v(x...) = collect(x)

  p_f_space   = map(v,f_p_space...)
  p_mspace    = map(f->MultiFieldFESpace(f;kwargs...),p_f_space)
  style       = PartitionedArrays.getany(map(MultiFieldStyle,p_mspace))
  gids        = generate_multi_field_gids(style,f_dspace,p_mspace)
  vector_type = _find_vector_type(p_mspace,gids;split_own_and_ghost=split_own_and_ghost)
  DistributedMultiFieldFESpace(f_dspace,p_mspace,gids,vector_type)
end

function generate_multi_field_gids(
  ::MultiFieldStyle,
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
  return gids
end

function generate_multi_field_gids(
  ::BlockMultiFieldStyle{NB,SB,P},
  f_dspace::Vector{<:DistributedSingleFieldFESpace},
  p_mspace::AbstractArray{<:MultiFieldFESpace}) where {NB,SB,P}

  block_ranges = MultiField.get_block_ranges(NB,SB,P)
  block_gids = map(block_ranges) do range
    space = (length(range) == 1) ? f_dspace[range[1]] : MultiFieldFESpace(f_dspace[range])
    get_free_dof_ids(space)
  end
  return BlockPRange(block_gids)
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

# BlockSparseMatrixAssemblers

const DistributedBlockSparseMatrixAssembler{NB,NV,SB,P} = 
  MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P,<:DistributedSparseMatrixAssembler}

function FESpaces.SparseMatrixAssembler(
  local_mat_type,
  local_vec_type,
  trial::DistributedMultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}},
  test::DistributedMultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}},
  par_strategy=SubAssembledRows()) where {NB,SB,P}

  block_idx  = CartesianIndices((NB,NB))
  block_rows = blocks(test.gids)
  block_cols = blocks(trial.gids)
  block_assemblers = map(block_idx) do idx
    rows = block_rows[idx[1]]; cols = block_cols[idx[2]]
    return SparseMatrixAssembler(local_mat_type,local_vec_type,rows,cols,par_strategy)
  end

  NV = length(P)
  return MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P}(block_assemblers)
end

function local_views(a::MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P}) where {NB,NV,SB,P}
  assems = a.block_assemblers
  array = to_parray_of_arrays(map(local_views,assems))
  return map(MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P},array)
end

function local_views(a::MatrixBlock,rows,cols)
  idx = CartesianIndices(axes(a))
  array = map(idx) do I
    local_views(a[I],rows[I[1]],cols[I[2]])
  end
  return map(b -> ArrayBlock(b,a.touched), to_parray_of_arrays(array))
end

function local_views(a::VectorBlock,rows)
  idx = CartesianIndices(axes(a))
  array = map(idx) do I
    local_views(a[I],rows[I])
  end
  return map(b -> ArrayBlock(b,a.touched), to_parray_of_arrays(array))
end

function local_views(a::ArrayBlockView,axes...)
  array = local_views(a.array,axes...)
  map(array) do array
    ArrayBlockView(array,a.block_map)
  end
end

# SparseMatrixAssembler API

function FESpaces.symbolic_loop_matrix!(A,a::DistributedBlockSparseMatrixAssembler,matdata)
  rows = get_rows(a)
  cols = get_cols(a)
  map(symbolic_loop_matrix!,local_views(A,rows,cols),local_views(a),matdata)
end

function FESpaces.numeric_loop_matrix!(A,a::DistributedBlockSparseMatrixAssembler,matdata)
  rows = get_rows(a)
  cols = get_cols(a)
  map(numeric_loop_matrix!,local_views(A,rows,cols),local_views(a),matdata)
end

function FESpaces.symbolic_loop_vector!(b,a::DistributedBlockSparseMatrixAssembler,vecdata)
  rows = get_rows(a)
  map(symbolic_loop_vector!,local_views(b,rows),local_views(a),vecdata)
end

function FESpaces.numeric_loop_vector!(b,a::DistributedBlockSparseMatrixAssembler,vecdata)
  rows = get_rows(a)
  map(numeric_loop_vector!,local_views(b,rows),local_views(a),vecdata)
end

function FESpaces.symbolic_loop_matrix_and_vector!(A,b,a::DistributedBlockSparseMatrixAssembler,data)
  rows = get_rows(a)
  cols = get_cols(a)
  map(symbolic_loop_matrix_and_vector!,local_views(A,rows,cols),local_views(b,rows),local_views(a),data)
end

function FESpaces.numeric_loop_matrix_and_vector!(A,b,a::DistributedBlockSparseMatrixAssembler,data)
  rows = get_rows(a)
  cols = get_cols(a)
  map(numeric_loop_matrix_and_vector!,local_views(A,rows,cols),local_views(b,rows),local_views(a),data)
end
