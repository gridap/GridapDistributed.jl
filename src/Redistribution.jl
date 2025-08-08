
# RedistributeGlue : Redistributing discrete models

"""
  RedistributeGlue

  Glue linking two distributions of the same mesh.
  
  - `new_parts`: Array with the new part IDs (and comms)
  - `old_parts`: Array with the old part IDs (and comms)
  - `parts_rcv`: Array with the part IDs from which each part receives
  - `parts_snd`: Array with the part IDs to which each part sends
  - `lids_rcv` : Local IDs of the entries that are received from each part
  - `lids_snd` : Local IDs of the entries that are sent to each part
  - `old2new`  : Mapping of local IDs from the old to the new mesh
  - `new2old`  : Mapping of local IDs from the new to the old mesh
"""
struct RedistributeGlue
  new_parts :: AbstractArray{<:Integer}
  old_parts :: AbstractArray{<:Integer}
  parts_rcv :: AbstractArray{<:AbstractVector{<:Integer}}
  parts_snd :: AbstractArray{<:AbstractVector{<:Integer}}
  lids_rcv  :: AbstractArray{<:JaggedArray{<:Integer}}
  lids_snd  :: AbstractArray{<:JaggedArray{<:Integer}}
  old2new   :: AbstractArray{<:AbstractVector{<:Integer}}
  new2old   :: AbstractArray{<:AbstractVector{<:Integer}}
end

get_parts(g::RedistributeGlue) = get_parts(g.parts_rcv)

function Base.reverse(g::RedistributeGlue)
  RedistributeGlue(
    g.old_parts,g.new_parts,
    g.parts_snd,g.parts_rcv,
    g.lids_snd,g.lids_rcv,
    g.new2old,g.old2new
  )
end

function get_old_and_new_parts(g::RedistributeGlue,reverse::Val{false})
  return g.old_parts, g.new_parts
end

function get_old_and_new_parts(g::RedistributeGlue,reverse::Val{true})
  return g.new_parts, g.old_parts
end

function get_glue_components(glue::RedistributeGlue,reverse::Val{false})
  return glue.lids_rcv, glue.lids_snd, glue.parts_rcv, glue.parts_snd, glue.new2old
end

function get_glue_components(glue::RedistributeGlue,reverse::Val{true})
  return glue.lids_snd, glue.lids_rcv, glue.parts_snd, glue.parts_rcv, glue.old2new
end

# Old Machinery
# Redistribution of cell-wise dofs, free values and FEFunctions

function _allocate_cell_wise_dofs(T,cell_to_ldofs)
  map(cell_to_ldofs) do cell_to_ldofs
    cache  = array_cache(cell_to_ldofs)
    ncells = length(cell_to_ldofs)
    ptrs   = Vector{Int32}(undef,ncells+1)
    for cell in 1:ncells
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      ptrs[cell+1] = length(ldofs)
    end
    PartitionedArrays.length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    data  = Vector{T}(undef,ndata)
    PartitionedArrays.JaggedArray(data,ptrs)
  end
end

function _update_cell_dof_values_with_local_info!(cell_dof_values_new,
                                                  cell_dof_values_old,
                                                  new2old)
  map(
    cell_dof_values_new,cell_dof_values_old,new2old
  ) do cell_dof_values_new,cell_dof_values_old,new2old
  ocache = array_cache(cell_dof_values_old)
  for (ncell,ocell) in enumerate(new2old)
    if ocell!=0
      # Copy ocell to ncell
      oentry = getindex!(ocache,cell_dof_values_old,ocell)
      range  = cell_dof_values_new.ptrs[ncell]:cell_dof_values_new.ptrs[ncell+1]-1
      cell_dof_values_new.data[range] .= oentry
    end
  end
  end
end

function _allocate_comm_data(T,num_dofs_x_cell,lids)
  map(num_dofs_x_cell,lids) do num_dofs_x_cell,lids
    n = length(lids)
    ptrs = Vector{Int32}(undef,n+1)
    ptrs.= 0
    for i = 1:n
      for j = lids.ptrs[i]:lids.ptrs[i+1]-1
        ptrs[i+1] = ptrs[i+1] + num_dofs_x_cell.data[j]
      end
    end
    PartitionedArrays.length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    data  = Vector{T}(undef,ndata)
    PartitionedArrays.JaggedArray(data,ptrs)
  end
end

function _pack_snd_data!(snd_data,cell_dof_values,snd_lids)
  map(snd_data,cell_dof_values,snd_lids) do snd_data,cell_dof_values,snd_lids
    cache = array_cache(cell_dof_values)
    s = 1
    for i = 1:length(snd_lids)
      for j = snd_lids.ptrs[i]:snd_lids.ptrs[i+1]-1
        cell  = snd_lids.data[j]
        ldofs = getindex!(cache,cell_dof_values,cell)

        e = s+length(ldofs)-1
        range = s:e
        snd_data.data[range] .= ldofs
        s = e+1
      end
    end
  end
end

function _unpack_rcv_data!(cell_dof_values,rcv_data,rcv_lids)
  map(cell_dof_values,rcv_data,rcv_lids) do cell_dof_values,rcv_data,rcv_lids
    s = 1
    for i = 1:length(rcv_lids.ptrs)-1
      for j = rcv_lids.ptrs[i]:rcv_lids.ptrs[i+1]-1
        cell = rcv_lids.data[j]
        range_cell_dof_values = cell_dof_values.ptrs[cell]:cell_dof_values.ptrs[cell+1]-1
        
        e = s+length(range_cell_dof_values)-1
        range_rcv_data = s:e
        cell_dof_values.data[range_cell_dof_values] .= rcv_data.data[range_rcv_data]
        s = e+1
      end
    end
  end
end

function _num_dofs_x_cell(cell_dofs_array,lids)
  map(cell_dofs_array,lids) do cell_dofs_array, lids
     data = [length(cell_dofs_array[i]) for i = 1:length(cell_dofs_array) ]
     PartitionedArrays.JaggedArray(data,lids.ptrs)
  end
end

function get_redistribute_cell_dofs_cache(
  cell_dof_values_old,
  cell_dof_ids_new,
  model_new,
  glue::RedistributeGlue;
  reverse=false,
  T = Float64
)
  lids_rcv, lids_snd, parts_rcv, parts_snd, new2old = get_glue_components(glue,Val(reverse))

  cell_dof_values_old = change_parts(cell_dof_values_old,get_parts(glue);default=[])
  cell_dof_ids_new    = change_parts(cell_dof_ids_new,get_parts(glue);default=[[]])

  num_dofs_x_cell_snd = _num_dofs_x_cell(cell_dof_values_old, lids_snd)
  num_dofs_x_cell_rcv = _num_dofs_x_cell(cell_dof_ids_new, lids_rcv)
  snd_data = _allocate_comm_data(T,num_dofs_x_cell_snd, lids_snd)
  rcv_data = _allocate_comm_data(T,num_dofs_x_cell_rcv, lids_rcv)

  cell_dof_values_new = _allocate_cell_wise_dofs(T,cell_dof_ids_new)

  caches = snd_data, rcv_data, cell_dof_values_new
  return caches
end

function redistribute_cell_dofs(
  cell_dof_values_old,
  cell_dof_ids_new,
  model_new,
  glue::RedistributeGlue;
  reverse=false,
  T = Float64
)
  caches = get_redistribute_cell_dofs_cache(cell_dof_values_old,cell_dof_ids_new,model_new,glue;reverse=reverse,T=T)
  return redistribute_cell_dofs!(caches,cell_dof_values_old,cell_dof_ids_new,model_new,glue;reverse=reverse)
end

function redistribute_cell_dofs!(
  caches,
  cell_dof_values_old,
  cell_dof_ids_new,
  model_new,
  glue::RedistributeGlue;
  reverse=false
)

  snd_data, rcv_data, cell_dof_values_new = caches
  lids_rcv, lids_snd, parts_rcv, parts_snd, new2old = get_glue_components(glue,Val(reverse))

  cell_dof_values_old = change_parts(cell_dof_values_old,get_parts(glue);default=[])
  cell_dof_ids_new    = change_parts(cell_dof_ids_new,get_parts(glue);default=[[]])

  _pack_snd_data!(snd_data,cell_dof_values_old,lids_snd)

  graph = ExchangeGraph(parts_snd,parts_rcv)
  t = exchange!(rcv_data,snd_data,graph)
  wait(t)

  # We have to build the owned part of "cell_dof_values_new" out of
  #  1. cell_dof_values_old (for those cells s.t. new2old[:]!=0)
  #  2. cell_dof_values_new_rcv (for those cells s.t. new2old[:]=0)
  _update_cell_dof_values_with_local_info!(
    cell_dof_values_new, cell_dof_values_old, new2old
  )
  _unpack_rcv_data!(cell_dof_values_new,rcv_data,lids_rcv)

  # Now that every part knows it's new owned dofs, exchange ghosts
  if !isnothing(model_new)
    new_parts = get_parts(model_new)
    cell_dof_values_new = change_parts(cell_dof_values_new,new_parts)
    cache = fetch_vector_ghost_values_cache(cell_dof_values_new,partition(get_cell_gids(model_new)))
    fetch_vector_ghost_values!(cell_dof_values_new,cache) |> wait
  end
  return cell_dof_values_new
end

function _get_cell_dof_ids_inner_space(s::FESpace)
  get_cell_dof_ids(s)
end 

function _get_cell_dof_ids_inner_space(s::FESpaceWithLinearConstraints)
  get_cell_dof_ids(s.space)
end

# Required in order to avoid returning the results of get_cell_dof_ids(space)
# in the case of a FESpaceWithLinearConstraints wrapped around a TrialFESpace
function _get_cell_dof_ids_inner_space(s::TrialFESpace)
  _get_cell_dof_ids_inner_space(s.space)
end

function redistribute_fe_function(
  uh_old::Union{DistributedSingleFieldFEFunction,Nothing},
  Uh_new::Union{DistributedSingleFieldFESpace,Nothing},
  model_new,
  glue::RedistributeGlue;
  reverse=false
)

  old_parts, new_parts = get_old_and_new_parts(glue,Val(reverse))
  cell_dof_values_old  = i_am_in(old_parts) ? map(get_cell_dof_values,local_views(uh_old)) : nothing
  cell_dof_ids_new     = i_am_in(new_parts) ? map(_get_cell_dof_ids_inner_space,local_views(Uh_new)) : nothing
  cell_dof_values_new  = redistribute_cell_dofs(cell_dof_values_old,cell_dof_ids_new,model_new,glue;reverse=reverse)

  # Assemble the new FEFunction
  if i_am_in(new_parts)
    free_values, dirichlet_values = Gridap.FESpaces.gather_free_and_dirichlet_values(Uh_new,cell_dof_values_new)
    free_values = PVector(free_values,partition(Uh_new.gids))
    uh_new = FEFunction(Uh_new,free_values,dirichlet_values)
    return uh_new
  else
    return nothing
  end
end

for T in [:DistributedSingleFieldFESpace,:DistributedMultiFieldFESpace]
  @eval begin
    _get_fe_type(::$T,::Nothing) = $T
    _get_fe_type(::Nothing,::$T) = $T
    _get_fe_type(::$T,::$T) = $T
  end
end

function redistribute_free_values(
  fv_new::Union{PVector,Nothing},
  Uh_new::Union{DistributedSingleFieldFESpace,DistributedMultiFieldFESpace,Nothing},
  fv_old::Union{PVector,Nothing},
  dv_old::Union{AbstractArray,Nothing},
  Uh_old::Union{DistributedSingleFieldFESpace,DistributedMultiFieldFESpace,Nothing},
  model_new,
  glue::RedistributeGlue;
  reverse=false
)
  caches = get_redistribute_free_values_cache(fv_new,Uh_new,fv_old,dv_old,Uh_old,model_new,glue;reverse=reverse)
  return redistribute_free_values!(caches,fv_new,Uh_new,fv_old,dv_old,Uh_old,model_new,glue;reverse=reverse)
end

function get_redistribute_free_values_cache(
  fv_new,Uh_new,
  fv_old,dv_old,Uh_old,
  model_new,glue::RedistributeGlue;
  reverse=false
)
  T = _get_fe_type(Uh_new,Uh_old)
  get_redistribute_free_values_cache(T,fv_new,Uh_new,fv_old,dv_old,Uh_old,model_new,glue;reverse=reverse)
end

function get_redistribute_free_values_cache(
  ::Type{DistributedSingleFieldFESpace},
  fv_new,Uh_new,
  fv_old,dv_old,Uh_old,
  model_new,glue::RedistributeGlue;
  reverse=false
)
  old_parts, new_parts = get_old_and_new_parts(glue,Val(reverse))
  cell_dof_values_old = i_am_in(old_parts) ? map(scatter_free_and_dirichlet_values,local_views(Uh_old),local_views(fv_old),dv_old) : nothing
  cell_dof_ids_new    = i_am_in(new_parts) ? map(_get_cell_dof_ids_inner_space, local_views(Uh_new)) : nothing
  caches = get_redistribute_cell_dofs_cache(cell_dof_values_old,cell_dof_ids_new,model_new,glue;reverse=reverse)
  return caches
end

function get_redistribute_free_values_cache(
  ::Type{DistributedMultiFieldFESpace},
  fv_new,Uh_new,
  fv_old,dv_old,Uh_old,
  model_new,glue::RedistributeGlue;
  reverse=false
)
  old_parts, new_parts = get_old_and_new_parts(glue,Val(reverse))

  if i_am_in(old_parts)
    Uh_old_i = Uh_old.field_fe_space
    fv_old_i = map(i->restrict_to_field(Uh_old,fv_old,i),1:num_fields(Uh_old))
    dv_old_i = dv_old
  else
    nfields = num_fields(Uh_new) # The other is not Nothing
    Uh_old_i = [nothing for i = 1:nfields]
    fv_old_i = [nothing for i = 1:nfields]
    dv_old_i = [nothing for i = 1:nfields]
  end

  if i_am_in(new_parts)
    Uh_new_i = Uh_new.field_fe_space
    fv_new_i = map(i->restrict_to_field(Uh_new,fv_new,i),1:num_fields(Uh_new))
  else
    nfields = num_fields(Uh_old) # The other is not Nothing
    Uh_new_i = [nothing for i = 1:nfields]
    fv_new_i = [nothing for i = 1:nfields]
  end

  caches = map(Uh_new_i,Uh_old_i,fv_new_i,fv_old_i,dv_old_i) do Uh_new_i,Uh_old_i,fv_new_i,fv_old_i,dv_old_i
    get_redistribute_free_values_cache(DistributedSingleFieldFESpace,fv_new_i,Uh_new_i,fv_old_i,dv_old_i,Uh_old_i,model_new,glue;reverse=reverse)
  end

  return caches
end

function redistribute_free_values!(
  caches,
  fv_new,Uh_new,
  fv_old,dv_old,Uh_old,
  model_new,
  glue::RedistributeGlue;
  reverse=false
)
  T = _get_fe_type(Uh_new,Uh_old)
  redistribute_free_values!(T,caches,fv_new,Uh_new,fv_old,dv_old,Uh_old,model_new,glue;reverse=reverse)
end

function redistribute_free_values!(
  ::Type{DistributedSingleFieldFESpace},
  caches,
  fv_new::Union{PVector,Nothing},
  Uh_new::Union{DistributedSingleFieldFESpace,Nothing},
  fv_old::Union{PVector,Nothing},
  dv_old::Union{AbstractArray,Nothing},
  Uh_old::Union{DistributedSingleFieldFESpace,Nothing},
  model_new,
  glue::RedistributeGlue;
  reverse=false
)
  old_parts, new_parts = get_old_and_new_parts(glue,Val(reverse))
  cell_dof_values_old = i_am_in(old_parts) ? map(scatter_free_and_dirichlet_values,local_views(Uh_old),local_views(fv_old),dv_old) : nothing
  cell_dof_ids_new    = i_am_in(new_parts) ? map(_get_cell_dof_ids_inner_space, local_views(Uh_new)) : nothing
  cell_dof_values_new = redistribute_cell_dofs!(caches,cell_dof_values_old,cell_dof_ids_new,model_new,glue;reverse=reverse)

  # Gather the new free dofs
  if i_am_in(new_parts)
    Gridap.FESpaces.gather_free_values!(fv_new,Uh_new,cell_dof_values_new)
  end
  return fv_new
end

function redistribute_free_values!(
  ::Type{DistributedMultiFieldFESpace},
  caches,
  fv_new::Union{PVector,Nothing},
  Uh_new::Union{DistributedMultiFieldFESpace,Nothing},
  fv_old::Union{PVector,Nothing},
  dv_old::Union{AbstractArray,Nothing},
  Uh_old::Union{DistributedMultiFieldFESpace,Nothing},
  model_new,
  glue::RedistributeGlue;
  reverse=false
)
  old_parts, new_parts = get_old_and_new_parts(glue,Val(reverse))

  if i_am_in(old_parts)
    Uh_old_i = Uh_old.field_fe_space
    fv_old_i = map(i->restrict_to_field(Uh_old,fv_old,i),1:num_fields(Uh_old))
    dv_old_i = dv_old
  else
    nfields = num_fields(Uh_new) # The other is not Nothing
    Uh_old_i = [nothing for i = 1:nfields]
    fv_old_i = [nothing for i = 1:nfields]
    dv_old_i = [nothing for i = 1:nfields]
  end

  if i_am_in(new_parts)
    Uh_new_i = Uh_new.field_fe_space
    fv_new_i = map(i->restrict_to_field(Uh_new,fv_new,i),1:num_fields(Uh_new))
  else
    nfields = num_fields(Uh_old) # The other is not Nothing
    Uh_new_i = [nothing for i = 1:nfields]
    fv_new_i = [nothing for i = 1:nfields]
  end

  map(Uh_new_i,Uh_old_i,fv_new_i,fv_old_i,dv_old_i,caches) do Uh_new_i,Uh_old_i,fv_new_i,fv_old_i,dv_old_i,caches
    redistribute_free_values!(DistributedSingleFieldFESpace,caches,fv_new_i,Uh_new_i,fv_old_i,dv_old_i,Uh_old_i,model_new,glue;reverse=reverse)
  end
end

############################################################################################
# New Machinery
# Allows redistribution of arbitrary PVectors, without the need of redistributing 
# things cell-wise.

"""
    redistribute_array_by_cells(
      old_lid_to_data, 
      old_cell_to_old_lid,
      new_cell_to_new_lid,
      model_new, glue;
      T = Int32, reverse=false,
    )

Redistributes an array using the old machinery. This is used to create the new index partitions
that the new machinery requires.

Takes in: 

- `old_lid_to_data`: The data to redistribute, indexed by local IDs in the old communicator.
- `old_cell_to_old_lid`: The mapping from cells to local data IDs in the old communicator.
- `new_cell_to_new_lid`: The mapping from cells to local data IDs in the new communicator.
- `model_new`: The model in the new communicator.
- `glue`: The `RedistributeGlue` glue.

Returns: 

- `new_lid_to_old_data`: The redistributed data, indexed by local IDs in the new communicator.

"""
function redistribute_array_by_cells(
  old_lid_to_data, 
  old_cell_to_old_lid,
  new_cell_to_new_lid,
  model_new, glue;
  T = Int32, reverse=false,
)
  if !isnothing(old_cell_to_old_lid) && !isnothing(old_lid_to_data)
    old_cell_to_old_data = map(old_cell_to_old_lid, old_lid_to_data) do old_cell_to_old_lid, old_lid_to_data
      jagged_array(view(old_lid_to_data,old_cell_to_old_lid.data), old_cell_to_old_lid.ptrs)
    end
  else
    old_cell_to_old_data = nothing
  end
  new_cell_to_old_data = redistribute_cell_dofs(
    old_cell_to_old_data, new_cell_to_new_lid, model_new, glue; T, reverse
  )
  if !isnothing(new_cell_to_new_lid) && !isnothing(new_cell_to_old_data)
    new_lid_to_old_data = map(new_cell_to_new_lid, new_cell_to_old_data) do new_cell_to_new_lid, new_cell_to_old_data
      new_lid_to_old_data = zeros(T, maximum(new_cell_to_new_lid.data;init=0))
      for (new_lid, old_data) in zip(new_cell_to_new_lid, new_cell_to_old_data)
        new_lid_to_old_data[new_lid] .= old_data
      end
      return new_lid_to_old_data
    end
  else
    new_lid_to_old_data = nothing
  end
  return new_lid_to_old_data
end

"""
    redistribute_indices(
      old_ids,
      old_cell_to_old_lid,
      new_cell_to_new_lid,
      model_new, glue;
      reverse=false,
    )

Redistributes an index partition from an old communicator to a new one, ensuring that 
the global ids coincide in both partitions.

Takes in:

- `old_ids`: The old index partition, providing the local-to-global ID map in the old communicator.
- `old_cell_to_old_lid`: The mapping from cells to local IDs in the old communicator.
- `new_cell_to_new_lid`: The mapping from cells to local IDs in the new communicator.
- `model_new`: The model in the new communicator.
- `glue`: The `RedistributeGlue` glue.

Returns:

- `old_ids`: The old index partition, but defined in the new communicator. It has empty 
             entries for the parts that are not in the old communicator.
- `red_old_ids`: The redistributed index partition, defined in the new communicator.

"""
function redistribute_indices(
  old_ids,
  old_cell_to_old_lid,
  new_cell_to_new_lid,
  model_new, glue;
  reverse=false,
)
  ranks = get_parts(glue)
  if !isnothing(old_ids)
    n_old = map(global_length,old_ids)
    old_lid_to_old_gid = map(local_to_global,old_ids)
    old_lid_to_old_owner = map(local_to_owner,old_ids)
  else
    n_old, old_lid_to_old_gid, old_lid_to_old_owner = nothing, nothing, nothing
  end

  new_lid_to_old_gid = change_parts(redistribute_array_by_cells(
    old_lid_to_old_gid, old_cell_to_old_lid, new_cell_to_new_lid, model_new, glue; T=Int,reverse,
  ), ranks; default=Int[])
  new_lid_to_old_owner = change_parts(redistribute_array_by_cells(
    old_lid_to_old_owner, old_cell_to_old_lid, new_cell_to_new_lid, model_new, glue; T=Int32,reverse,
  ), ranks; default=Int32[])

  n_old = emit(change_parts(n_old,ranks;default=0))
  old_lid_to_old_gid = change_parts(old_lid_to_old_gid, ranks; default=Int[])
  old_lid_to_old_owner = change_parts(old_lid_to_old_owner, ranks; default=Int32[])

  old_ids_bis = map(LocalIndices,n_old,ranks,old_lid_to_old_gid,old_lid_to_old_owner)
  red_old_ids = map(LocalIndices,n_old,ranks,new_lid_to_old_gid,new_lid_to_old_owner)
  return old_ids_bis, red_old_ids
end

function redistribution_neighbors(indices, indices_red)
  nbors_rcv, nbors_snd = assembly_neighbors(indices_red)
  return nbors_snd, nbors_rcv
end

"""
    redistribution_local_indices(indices, indices_red) -> (lids_snd, lids_rcv)

Returns the local indices to be communicated when redistributing from `indices` to `indices_red`.

CAREFUL: Unlike for the assembly operation, these snd/rcv indices are NOT symmetric.

"""
function redistribution_local_indices(indices, indices_red)
  nbors_snd, nbors_rcv = redistribution_neighbors(indices, indices_red)
  redistribution_local_indices(indices, indices_red, nbors_snd, nbors_rcv)
end

function redistribution_local_indices(indices, indices_red, nbors_snd, nbors_rcv)

  lids_rcv, gids_rcv = map(indices_red, nbors_rcv) do indices_red, nbors_rcv
    rank = part_id(indices_red)
    
    owner_to_i = Dict(( owner => i for (i,owner) in enumerate(nbors_rcv) ))
    ptrs = zeros(Int32,length(nbors_rcv)+1)
    for owner in local_to_owner(indices_red)
      if owner != rank
        ptrs[owner_to_i[owner]+1] += 1
      end
    end
    Arrays.length_to_ptrs!(ptrs)

    data_lids = zeros(Int32,ptrs[end]-1)
    data_gids = zeros(Int,ptrs[end]-1)
    lid_to_gid = local_to_global(indices_red)
    for (lid,owner) in enumerate(local_to_owner(indices_red))
      if owner != rank
        p = ptrs[owner_to_i[owner]]
        data_lids[p] = lid
        data_gids[p] = lid_to_gid[lid]
        ptrs[owner_to_i[owner]] += 1
      end
    end
    Arrays.rewind_ptrs!(ptrs)

    return JaggedArray(data_lids, ptrs), JaggedArray(data_gids, ptrs)
  end |> tuple_of_arrays

  graph = ExchangeGraph(nbors_rcv,nbors_snd)
  gids_snd = PartitionedArrays.exchange_fetch(gids_rcv, graph)

  lids_snd = map(indices, gids_snd) do indices, gids_snd
    ptrs = gids_snd.ptrs
    data = zeros(Int32,ptrs[end]-1)
    gid_to_lid = global_to_local(indices)
    for (k,gid) in enumerate(gids_snd.data)
      data[k] = gid_to_lid[gid]
    end
    return JaggedArray(data, ptrs)
  end

  return lids_snd, lids_rcv
end

function p_vector_redistribution_cache(values, indices, indices_red)
  nbors_snd, nbors_rcv = redistribution_neighbors(indices, indices_red)
  lids_snd, lids_rcv = redistribution_local_indices(indices, indices_red, nbors_snd, nbors_rcv)
  buffers_snd, buffers_rcv = map(
    PartitionedArrays.assembly_buffers,values,lids_snd,lids_rcv
  ) |> tuple_of_arrays
  caches = map(PartitionedArrays.VectorAssemblyCache,nbors_snd, nbors_rcv,lids_snd, lids_rcv,buffers_snd,buffers_rcv)
  return caches
end

"""
    redistribute(v::PVector,new_indices)
    redistribute!(w::PVector,v::PVector,cache)

Redistributes a PVector `v` to a new partition defined by `new_indices`.
"""
function redistribute(v::PVector,new_indices)
  indices = partition(axes(v,1))
  cache = p_vector_redistribution_cache(partition(v), indices, new_indices)
  w = pzeros(new_indices)
  return redistribute!(w, v, cache)
end

function redistribute!(w::PVector,v::PVector,cache)
  function setup_snd(values, cache)
    cache.buffer_snd.data .= view(values,cache.local_indices_snd.data)
    return cache.buffer_rcv, cache.buffer_snd, cache.neighbors_rcv, cache.neighbors_snd
  end
  function copy_owned(values_red, values, indices_red, indices)
    lid_to_gid_red = local_to_global(indices_red)
    lid_to_own_red = local_to_own(indices_red)
    gid_to_lid = global_to_local(indices)
    for (lid_red, gid) in enumerate(lid_to_gid_red)
      if !iszero(lid_to_own_red[lid_red])
        lid = gid_to_lid[gid]
        @assert !iszero(lid)
        values_red[lid_red] = values[lid]
      end
    end
  end
  function copy_rcv(values_red, cache)
    view(values_red, cache.local_indices_rcv.data) .= cache.buffer_rcv.data
  end

  values, indices = partition(v), partition(axes(v,1))
  values_red, indices_red = partition(w), partition(axes(w,1))
  buffer_rcv, buffer_snd, nbors_rcv, nbors_snd = map(setup_snd, values, cache) |> tuple_of_arrays
  graph = ExchangeGraph(nbors_snd, nbors_rcv)
  t = PartitionedArrays.exchange!(buffer_rcv, buffer_snd, graph)
  @async begin
    map(copy_owned, values_red, values, indices_red, indices)
    wait(t)
    map(copy_rcv, values_red, cache)
    return w
  end
end
