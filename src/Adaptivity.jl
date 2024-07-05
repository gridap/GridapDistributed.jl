
# DistributedAdaptedDiscreteModels

const DistributedAdaptedDiscreteModel{Dc,Dp} = GenericDistributedDiscreteModel{Dc,Dp,<:AbstractArray{<:AdaptedDiscreteModel{Dc,Dp}}}

function DistributedAdaptedDiscreteModel(
  model  :: DistributedDiscreteModel,
  parent :: DistributedDiscreteModel,
  glue   :: AbstractArray{<:AdaptivityGlue};
)
  models = map(local_views(model),local_views(parent),glue) do model, parent, glue
    AdaptedDiscreteModel(model,parent,glue)
  end
  gids = get_cell_gids(model)
  metadata = model.metadata
  return GenericDistributedDiscreteModel(models,gids;metadata)
end

function Adaptivity.get_model(model::DistributedAdaptedDiscreteModel)
  GenericDistributedDiscreteModel(
    map(get_model,local_views(model)),
    get_cell_gids(model);
    metadata=model.metadata
  )
end

function Adaptivity.get_parent(model::DistributedAdaptedDiscreteModel)
  msg = " Error: Cannot get global parent model. \n 
          We do not keep the global ids of the parent model within the children.\n
          You can extract the local parents with map(get_parent,local_views(model))"
  @notimplemented msg
end

function Adaptivity.get_adaptivity_glue(model::DistributedAdaptedDiscreteModel)
  return map(Adaptivity.get_adaptivity_glue,local_views(model))
end

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

function allocate_rcv_buffer(t::Type{T},g::RedistributeGlue) where T
  ptrs = local_indices_rcv.ptrs
  data = zeros(T,ptrs[end]-1)
  JaggedArray(data,ptrs)
end 

function allocate_snd_buffer(t::Type{T},g::RedistributeGlue) where T
  ptrs = local_indices_snd.ptrs
  data = zeros(T,ptrs[end]-1)
  JaggedArray(data,ptrs)
end

"""
  Redistributes an DistributedDiscreteModel to optimally 
  rebalance the loads between the processors. 
  Returns the rebalanced model and a RedistributeGlue instance. 
"""
function redistribute(::DistributedDiscreteModel,args...;kwargs...)
  @abstractmethod
end

function redistribute(model::DistributedAdaptedDiscreteModel,args...;kwargs...)
  # Local cmodels are AdaptedDiscreteModels. To correctly dispatch, we need to
  # extract the underlying models, then redistribute.
  _model = GenericDistributedDiscreteModel(
    map(get_model,local_views(model)),
    get_cell_gids(model);
    metadata=model.metadata
  )
  return redistribute(_model,args...;kwargs...)
end

# Redistribution of cell-wise dofs, free values and FEFunctions

function _allocate_cell_wise_dofs(cell_to_ldofs)
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
    data  = Vector{Float64}(undef,ndata)
    PartitionedArrays.JaggedArray(data,ptrs)
  end
end

function _update_cell_dof_values_with_local_info!(cell_dof_values_new,
                                                  cell_dof_values_old,
                                                  new2old)
   map(cell_dof_values_new,
       cell_dof_values_old,
       new2old) do cell_dof_values_new,cell_dof_values_old,new2old
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

function _allocate_comm_data(num_dofs_x_cell,lids)
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
    data  = Vector{Float64}(undef,ndata)
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
  reverse=false
)
  lids_rcv, lids_snd, parts_rcv, parts_snd, new2old = get_glue_components(glue,Val(reverse))

  cell_dof_values_old = change_parts(cell_dof_values_old,get_parts(glue);default=[])
  cell_dof_ids_new    = change_parts(cell_dof_ids_new,get_parts(glue);default=[[]])

  num_dofs_x_cell_snd = _num_dofs_x_cell(cell_dof_values_old, lids_snd)
  num_dofs_x_cell_rcv = _num_dofs_x_cell(cell_dof_ids_new, lids_rcv)
  snd_data = _allocate_comm_data(num_dofs_x_cell_snd, lids_snd)
  rcv_data = _allocate_comm_data(num_dofs_x_cell_rcv, lids_rcv)

  cell_dof_values_new = _allocate_cell_wise_dofs(cell_dof_ids_new)

  caches = snd_data, rcv_data, cell_dof_values_new
  return caches
end

function redistribute_cell_dofs(
  cell_dof_values_old,
  cell_dof_ids_new,
  model_new,
  glue::RedistributeGlue;
  reverse=false
)
  caches = get_redistribute_cell_dofs_cache(cell_dof_values_old,cell_dof_ids_new,model_new,glue;reverse=reverse)
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


# Cartesian Model uniform refinement

function Adaptivity.refine(
  cmodel::DistributedDiscreteModel{Dc},
  refs::Integer = 2
) where Dc
  Adaptivity.refine(cmodel,Tuple(fill(refs,Dc)))
end

function Adaptivity.refine(
  cmodel::DistributedAdaptedDiscreteModel{Dc},
  refs::NTuple{Dc,<:Integer}
) where Dc

  # Local cmodels are AdaptedDiscreteModels. To correctly dispatch, we need to
  # extract the underlying models, then refine.
  _cmodel = get_model(cmodel)
  _fmodel = refine(_cmodel,refs)

  # Now the issue is that the local parents are not pointing to local_views(cmodel).
  # We have to fix that...
  fmodel = GenericDistributedDiscreteModel(
    map(get_model,local_views(_fmodel)),
    get_cell_gids(_fmodel);
    metadata=_fmodel.metadata
  )
  glues = get_adaptivity_glue(_fmodel)
  return DistributedAdaptedDiscreteModel(fmodel,cmodel,glues)
end

function Adaptivity.refine(
  cmodel::DistributedCartesianDiscreteModel{Dc},
  refs::NTuple{Dc,<:Integer},
) where Dc

  ranks = linear_indices(local_views(cmodel))
  desc, parts = cmodel.metadata.descriptor, cmodel.metadata.mesh_partition

  # Create the new model
  nC = desc.partition
  domain = Adaptivity._get_cartesian_domain(desc)
  nF = nC .* refs
  fmodel = CartesianDiscreteModel(
    ranks,parts,domain,nF;map=desc.map,isperiodic=desc.isperiodic
  )

  # The idea for the glue is the following: 
  #   For each coarse local model (owned + ghost), we can use the serial code to create
  #   the glue. However, this glue is NOT fully correct. 
  #   Why? Because all the children belonging to coarse ghost cells are in the glue. This 
  #   is not correct, since we only want to keep the children which are ghosts in the new model.
  #   To this end, we have to remove the extra fine layers of ghosts from the glue. This we 
  #   can do thanks to how predictable the Cartesian model is.
  glues = map(ranks,local_views(cmodel)) do rank,cmodel
    # Glue for the local models, of size nC_local .* ref
    desc_local = get_cartesian_descriptor(cmodel)
    nC_local = desc_local.partition
    nF_local = nC_local .* refs
    f2c_map, child_map = Adaptivity._create_cartesian_f2c_maps(nC_local,refs)
    
    # Remove extra fine layers of ghosts
    p = Tuple(CartesianIndices(parts)[rank])
    periodic_ghosts = map((isp,np) -> (np==1) ? false : isp, desc.isperiodic, parts)
    local_range = map(p,parts,periodic_ghosts,nF_local,refs) do p, np, pg, nFl, refs
      has_ghost_start = (np > 1) && ((p != 1)  || pg)
      has_ghost_stop  = (np > 1) && ((p != np) || pg)
      # If has coarse ghost layer, remove all fine layers but one at each end
      start = 1 + has_ghost_start*(refs-1)
      stop  = nFl - has_ghost_stop*(refs-1)
      return start:stop
    end

    _indices = LinearIndices(nF_local)[local_range...]
    indices = reshape(_indices,length(_indices))
    f2c_cell_map = f2c_map[indices]
    fcell_to_child_id = child_map[indices]
  
    # Create the glue
    faces_map = [(d==Dc) ? f2c_cell_map : Int[] for d in 0:Dc]
    poly   = (Dc == 2) ? QUAD : HEX
    reffe  = LagrangianRefFE(Float64,poly,1)
    rrules = RefinementRule(reffe,refs)
    return AdaptivityGlue(faces_map,fcell_to_child_id,rrules)
  end

  # Finally, we need to propagate the face labelings to the new model,
  # and create the local adapted models.
  fmodels = map(local_views(fmodel),local_views(cmodel),glues) do fmodel, cmodel, glue
    # Propagate face labels
    clabels = get_face_labeling(cmodel)
    ctopo   = get_grid_topology(cmodel)
    ftopo   = get_grid_topology(fmodel)
    flabels = Adaptivity._refine_face_labeling(clabels,glue,ctopo,ftopo)

    _fmodel = CartesianDiscreteModel(get_grid(fmodel),ftopo,flabels)
    return AdaptedDiscreteModel(_fmodel,cmodel,glue)
  end

  fgids = get_cell_gids(fmodel)
  metadata = fmodel.metadata
  return GenericDistributedDiscreteModel(fmodels,fgids;metadata)
end

# Cartesian Model redistribution

@inline function redistribute(  
  old_model::Union{DistributedCartesianDiscreteModel,Nothing},
  pdesc::DistributedCartesianDescriptor;
  old_ranks = nothing
)
  redistribute_cartesian(old_model,pdesc;old_ranks)
end

"""
    redistribute_cartesian(old_model,new_ranks,new_parts)
    redistribute_cartesian(old_model,pdesc::DistributedCartesianDescriptor)
  
  Redistributes a DistributedCartesianDiscreteModel to a new set of ranks and parts.
  Only redistributes into a superset of the old_model ranks (i.e. towards more processors).
"""
function redistribute_cartesian(
  old_model::Union{DistributedCartesianDiscreteModel,Nothing},
  new_ranks,
  new_parts;
  old_ranks = nothing
)
  _pdesc = isnothing(old_model) ? nothing : old_model.metadata
  pdesc  = emit_cartesian_descriptor(_pdesc,new_ranks,new_parts)
  redistribute_cartesian(old_model,pdesc;old_ranks)
end

function redistribute_cartesian(
  old_model::Union{DistributedCartesianDiscreteModel,Nothing},
  pdesc::DistributedCartesianDescriptor{Dc};
  old_ranks = nothing
) where Dc
  new_ranks = pdesc.ranks
  new_parts = pdesc.mesh_partition

  desc = pdesc.descriptor
  ncells = desc.partition
  domain = Adaptivity._get_cartesian_domain(desc)
  _new_model = CartesianDiscreteModel(new_ranks,new_parts,domain,ncells)

  rglue = get_cartesian_redistribute_glue(_new_model,old_model;old_ranks)

  # Propagate face labelings to the new model
  new_labels = get_redistributed_face_labeling(_new_model,old_model,rglue)
  new_models = map(local_views(_new_model),local_views(new_labels)) do _new_model, new_labels
    CartesianDiscreteModel(get_grid(_new_model),get_grid_topology(_new_model),new_labels)
  end
  new_model = GenericDistributedDiscreteModel(
    new_models,get_cell_gids(_new_model);metadata=_new_model.metadata
  )

  return new_model, rglue
end

function get_cartesian_owners(gids,nparts,ncells)
  # This is currently O(sqrt(np)), but I believe we could make it 
  # O(ln(np)) if we ensured the search is sorted. Even faster if we sort 
  # the gids first, which progressively reduces the number of ranges to search.
  ranges = map(nparts,ncells) do np, nc
    map(p -> PartitionedArrays.local_range(p,np,nc,false,false), 1:np)
  end
  cart_ids = CartesianIndices(ncells)
  owner_ids = LinearIndices(nparts)
  owners = map(gids) do gid
    gid_cart = Tuple(cart_ids[gid])
    owner_cart = map((r,g) -> findfirst(ri -> g ∈ ri,r),ranges,gid_cart)
    return owner_ids[owner_cart...]
  end
  return owners
end

function get_cartesian_redistribute_glue(
  new_model::DistributedCartesianDiscreteModel{Dc},
  old_model::Union{DistributedCartesianDiscreteModel{Dc},Nothing};
  old_ranks = nothing
) where Dc
  pdesc = new_model.metadata
  desc  = pdesc.descriptor
  ncells = desc.partition

  # Components in the new partition
  new_ranks = pdesc.ranks
  new_parts = new_model.metadata.mesh_partition
  new_ids = partition(get_cell_gids(new_model))
  new_models = local_views(new_model)

  # Components in the old partition (if present)
  _old_parts = map(new_ranks) do r
    (r == 1) ? Int[old_model.metadata.mesh_partition...] : Int[]
  end
  old_parts = Tuple(PartitionedArrays.getany(emit(_old_parts)))
  _old_ids = isnothing(old_model) ? nothing : partition(get_cell_gids(old_model))
  old_ids = change_parts(_old_ids,new_ranks)
  _old_models = isnothing(old_model) ? nothing : local_views(old_model)
  old_models = change_parts(_old_models,new_ranks)

  # Produce the glue components
  old2new,new2old,parts_rcv,parts_snd,lids_rcv,lids_snd = map(
    new_ranks,new_models,old_models,new_ids,old_ids) do r, new_model, old_model, new_ids, old_ids

    if !isnothing(old_ids)
      # If I'm in the old subprocessor,
      #   - I send all owned old cells that I don't own in the new model.
      #   - I receive all owned new cells that I don't own in the old model.
      old2new = replace(find_local_to_local_map(old_ids,new_ids), -1 => 0)
      new2old = replace(find_local_to_local_map(new_ids,old_ids), -1 => 0)

      new_l2o = local_to_own(new_ids)
      old_l2o = local_to_own(old_ids)
      mask_rcv = map(1:local_length(new_ids)) do new_lid
        old_lid = new2old[new_lid]
        A = !iszero(new_l2o[new_lid]) # I own this cell in the new model
        B = (iszero(old_lid) || iszero(old_l2o[old_lid])) # I don't own this cell in the old model
        return A && B
      end
      ids_rcv = findall(mask_rcv)

      mask_snd = map(1:local_length(old_ids)) do old_lid
        new_lid = old2new[old_lid]
        A = !iszero(old_l2o[old_lid]) # I own this cell in the old model
        B = (iszero(new_lid) || iszero(new_l2o[new_lid])) # I don't own this cell in the new model
        return A && B
      end
      ids_snd = findall(mask_snd)
    else
      # If I'm not in the old subprocessor, 
      #   - I don't send anything. 
      #   - I receive all my owned cells in the new model.
      old2new = Int[]
      new2old = fill(0,num_cells(new_model))
      ids_rcv = collect(own_to_local(new_ids))
      ids_snd = Int[]
    end

    # When snd/rcv ids have been selected, we need to find their owners and prepare 
    # the snd/rcv buffers.

    # First, everyone can potentially receive stuff: 
    to_global!(ids_rcv,new_ids)
    ids_rcv_to_part = get_cartesian_owners(ids_rcv,old_parts,ncells)
    to_local!(ids_rcv,new_ids)
    parts_rcv = unique(ids_rcv_to_part)
    lids_rcv = map(parts_rcv) do nbor
      ids_rcv[findall(x -> x == nbor, ids_rcv_to_part)]
    end
    lids_rcv = convert(Vector{Vector{Int}}, lids_rcv)

    # Then, only the old subprocessor can potentially send stuff:
    if !isnothing(old_ids)
      to_global!(ids_snd,old_ids)
      ids_snd_to_part = get_cartesian_owners(ids_snd,new_parts,ncells)
      to_local!(ids_snd,old_ids)
      parts_snd = unique(ids_snd_to_part)
      lids_snd = map(parts_snd) do nbor
        ids_snd[findall(x -> x == nbor, ids_snd_to_part)]
      end
      lids_snd = convert(Vector{Vector{Int}}, lids_snd)
    else
      parts_snd = Int[]
      lids_snd  = [Int[]]
    end

    return old2new, new2old, parts_rcv, parts_snd, JaggedArray(lids_rcv), JaggedArray(lids_snd)
  end |> tuple_of_arrays

  # WARNING: This will fail if compared (===) with get_parts(old_model)
  # Do we really require this in the glue? Could we remove the old ranks?
  if isnothing(old_ranks)
    old_ranks = generate_subparts(new_ranks,prod(old_parts))
  end

  return RedistributeGlue(new_ranks,old_ranks,parts_rcv,parts_snd,lids_rcv,lids_snd,old2new,new2old)
end

function get_redistributed_face_labeling(
  new_model::DistributedCartesianDiscreteModel{Dc},
  old_model::Union{DistributedCartesianDiscreteModel{Dc},Nothing},
  glue::RedistributeGlue
) where Dc

  new_ranks = get_parts(new_model)

  _old_models = !isnothing(old_model) ? local_views(old_model) : nothing
  old_models = change_parts(_old_models,new_ranks)
  new_models = local_views(new_model)
  
  # Communicate facet entities
  new_d_to_dface_to_entity = map(new_models) do new_model
    Vector{Vector{Int32}}(undef,Dc+1)
  end

  for Df in 0:Dc

    # Pack entity data
    old_cell_to_face_entity, new_cell_to_face_entity = map(old_models,new_models) do old_model, new_model

      if !isnothing(old_model)
        old_labels = get_face_labeling(old_model)
        old_topo = get_grid_topology(old_model)

        old_cell2face = Geometry.get_faces(old_topo,Dc,Df)
        old_face2entity = old_labels.d_to_dface_to_entity[Df+1]

        old_cell_to_face_entity = Table(
          lazy_map(Reindex(old_face2entity),old_cell2face.data),
          old_cell2face.ptrs
        )
      end

      new_topo = get_grid_topology(new_model)
      new_cell2face = Geometry.get_faces(new_topo,Dc,Df)
      new_cell_to_face_entity = Table(
        zeros(eltype(new_cell2face.data),length(new_cell2face.data)),
        new_cell2face.ptrs
      )

      return old_cell_to_face_entity, new_cell_to_face_entity
    end |> tuple_of_arrays

    # Redistribute entity data
    redistribute_cell_dofs(
      old_cell_to_face_entity,new_cell_to_face_entity,new_model,glue
    )

    # Unpack entity data
    new_face2entity = map(new_models,new_cell_to_face_entity) do new_model,new_cell_to_face_entity
      new_topo = get_grid_topology(new_model)
      new_cell2face = Geometry.get_faces(new_topo,Dc,Df)
      
      new_face2entity = zeros(eltype(new_cell2face.data),Geometry.num_faces(new_topo,Df))
      for cell in 1:length(new_cell2face.ptrs)-1
        for pos in new_cell2face.ptrs[cell]:new_cell2face.ptrs[cell+1]-1
          face = new_cell2face.data[pos]
          new_face2entity[face] = new_cell_to_face_entity.data[pos]
        end
      end

      return new_face2entity
    end

    map(new_d_to_dface_to_entity,new_face2entity) do new_d_to_dface_to_entity,new_face2entity
      new_d_to_dface_to_entity[Df+1] = new_face2entity
    end
  end

  # Communicate entity tags
  old_tag_to_name, old_tag_to_entities = map(new_models) do new_model
    if !isnothing(old_model)
      new_labels = get_face_labeling(new_model)
      return new_labels.tag_to_name, new_labels.tag_to_entities
    else
      return String[], Vector{Int32}[]
    end
  end |> tuple_of_arrays
  new_tag_to_name = emit(old_tag_to_name)
  new_tag_to_entities = emit(old_tag_to_entities)

  new_labels = map(FaceLabeling,new_d_to_dface_to_entity,new_tag_to_entities,new_tag_to_name)
  return DistributedFaceLabeling(new_labels)
end
