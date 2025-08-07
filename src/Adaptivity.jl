
# DistributedAdaptedDiscreteModels

const DistributedAdaptedDiscreteModel{Dc,Dp} = GenericDistributedDiscreteModel{Dc,Dp,<:AbstractArray{<:AdaptedDiscreteModel{Dc,Dp}}}

struct DistributedAdaptedDiscreteModelCache{A,B,C}
  model_metadata::A
  parent_metadata::B
  parent_gids::C
end

function DistributedAdaptedDiscreteModel(
  model  :: DistributedDiscreteModel,
  parent :: DistributedDiscreteModel,
  glue   :: AbstractArray{<:AdaptivityGlue};
)
  models = map(local_views(model),local_views(parent),glue) do model, parent, glue
    AdaptedDiscreteModel(model,parent,glue)
  end
  gids = get_cell_gids(model)
  metadata = DistributedAdaptedDiscreteModelCache(
    model.metadata,parent.metadata,get_cell_gids(parent)
  )
  return GenericDistributedDiscreteModel(models,gids;metadata)
end

function Adaptivity.get_model(model::DistributedAdaptedDiscreteModel)
  GenericDistributedDiscreteModel(
    map(get_model,local_views(model)),
    get_cell_gids(model);
    metadata = model.metadata.model_metadata
  )
end

function Adaptivity.get_parent(model::DistributedAdaptedDiscreteModel)
  GenericDistributedDiscreteModel(
    map(get_parent,local_views(model)),
    model.metadata.parent_gids;
    metadata = model.metadata.parent_metadata
  )
end

function Adaptivity.get_adaptivity_glue(model::DistributedAdaptedDiscreteModel)
  return map(Adaptivity.get_adaptivity_glue,local_views(model))
end

function Adaptivity.is_child(m1::DistributedDiscreteModel,m2::DistributedDiscreteModel)
  reduce(&,map(Adaptivity.is_child,local_views(m1),local_views(m2)))
end

function Adaptivity.refine(
  cmodel::DistributedAdaptedDiscreteModel{Dc},args...;kwargs...
) where Dc
  # Local cmodels are AdaptedDiscreteModels. To correctly dispatch, we need to
  # extract the underlying models, then refine.
  _cmodel = get_model(cmodel)
  _fmodel = refine(_cmodel,args...;kwargs...)

  # Now the issue is that the local parents are not pointing to local_views(cmodel).
  # We have to fix that...
  fmodel = GenericDistributedDiscreteModel(
    map(get_model,local_views(_fmodel)),
    get_cell_gids(_fmodel);
    metadata=_fmodel.metadata.model_metadata
  )
  glues = get_adaptivity_glue(_fmodel)
  return DistributedAdaptedDiscreteModel(fmodel,cmodel,glues)
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
  _model = get_model(model)
  return redistribute(_model,args...;kwargs...)
end

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
  reverse=false
)
  lids_rcv, lids_snd, parts_rcv, parts_snd, new2old = get_glue_components(glue,Val(reverse))

  cell_dof_values_old = change_parts(cell_dof_values_old,get_parts(glue);default=[])
  cell_dof_ids_new    = change_parts(cell_dof_ids_new,get_parts(glue);default=[[]])

  T = eltype(eltype(cell_dof_values_old))
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
  cmodel::DistributedCartesianDiscreteModel{Dc},
  refs::Integer = 2
) where Dc
  Adaptivity.refine(cmodel,Tuple(fill(refs,Dc)))
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

  map_main(ranks) do r
    @debug " Refining DistributedCartesianModel:
      > Parent: $(repr("text/plain",cmodel.metadata)) 
      > Child:  $(repr("text/plain",fmodel.metadata))
    "
  end

  # The idea for the glue is the following: 
  #   For each coarse local model (owned + ghost), we can use the serial code to create
  #   the glue. However, this glue is NOT fully correct. 
  #   Why? Because all the children belonging to coarse ghost cells are in the glue. This 
  #   is not correct, since we only want to keep the children which are ghosts in the new model.
  #   To this end, we have to remove the extra fine layers of ghosts from the glue. This we 
  #   can do thanks to how predictable the Cartesian model is.
  glues = map(ranks,local_views(cmodel),local_views(fmodel)) do rank,cmodel,fmodel
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
    @debug "[$(rank)] nC_local=$(nC_local), nF_local = $(nF_local), refs=$(refs), periodic_ghosts=$(periodic_ghosts), local_range=$(local_range) \n"
    _nF = get_cartesian_descriptor(fmodel).partition
    @check all(map((n,r) -> n == length(r),_nF,local_range))

    _indices = LinearIndices(nF_local)[local_range...]
    indices = reshape(_indices,length(_indices))
    f2c_cell_map = f2c_map[indices]
    fcell_to_child_id = child_map[indices]
  
    # Create the glue
    faces_map = [(d==Dc) ? f2c_cell_map : Int[] for d in 0:Dc]
    poly   = (Dc == 2) ? QUAD : HEX
    reffe  = LagrangianRefFE(Float64,poly,1)
    rrules = Fill(RefinementRule(reffe,refs),num_cells(cmodel))
    return AdaptivityGlue(faces_map,fcell_to_child_id,rrules)
  end

  # Finally, we need to propagate the face labelings to the new model,
  # and create the local adapted models.
  fmodels = map(local_views(fmodel),local_views(cmodel),glues) do fmodel, cmodel, glue
    # Propagate face labels
    clabels = get_face_labeling(cmodel)
    ctopo   = get_grid_topology(cmodel)
    ftopo   = get_grid_topology(fmodel)
    flabels = Adaptivity.refine_face_labeling(clabels,glue,ctopo,ftopo)

    _fmodel = CartesianDiscreteModel(get_grid(fmodel),ftopo,flabels)
    return AdaptedDiscreteModel(_fmodel,cmodel,glue)
  end

  fgids = get_cell_gids(fmodel)
  metadata = DistributedAdaptedDiscreteModelCache(
    fmodel.metadata,cmodel.metadata,get_cell_gids(cmodel)
  )
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

  map_main(new_ranks) do r
    @debug "Redistributing DistributedCartesianModel:
      > Old: $(repr("text/plain",old_model.metadata))
      > New: $(repr("text/plain",_new_model.metadata))
    "
    msg1 = "Both models should have the same number of cells for redistribution!"
    @check old_model.metadata.descriptor.partition == ncells msg1
    msg2 = "Only redistribution to a higher number of processors is supported!"
    @check prod(old_model.metadata.mesh_partition) <= prod(new_parts) msg2
  end

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

  map_main(new_ranks) do r
    @debug "Creating RedistributeGlue:
      > Old: $(repr("text/plain",old_model.metadata))
      > New: $(repr("text/plain",new_model.metadata))
    "
  end

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
      new2old = fill(zero(Int),num_cells(new_model))
      ids_rcv = collect(Int,own_to_local(new_ids))
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
    @debug "[$(r)] parts_snd=$(parts_snd), parts_rcv=$(parts_rcv), n_lids_snd=$(map(length,lids_snd))), n_lids_rcv=$(map(length,lids_rcv))) \n"

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
  map_main(new_ranks) do r
    @debug "Redistributing face labeling:
      > Old: $(repr("text/plain",old_model.metadata))
      > New: $(repr("text/plain",new_model.metadata))
    "
  end

  _old_models = !isnothing(old_model) ? local_views(old_model) : nothing
  old_models = change_parts(_old_models,new_ranks)
  new_models = local_views(new_model)
  
  # Communicate facet entities
  new_d_to_dface_to_entity = map(new_models) do new_model
    Vector{Vector{Int32}}(undef,Dc+1)
  end

  for Df in 0:Dc

    # Pack entity data
    old_cell_to_face_entity, new_cell_to_face_ids = map(old_models,new_models) do old_model, new_model

      if !isnothing(old_model)
        old_labels = get_face_labeling(old_model)
        old_topo = get_grid_topology(old_model)

        old_cell2face = Geometry.get_faces(old_topo,Dc,Df)
        old_face2entity = old_labels.d_to_dface_to_entity[Df+1]

        old_cell_to_face_entity = Table(
          collect(Int32,lazy_map(Reindex(old_face2entity),old_cell2face.data)), # Avoid if possible
          old_cell2face.ptrs
        )
      else
        old_cell_to_face_entity = Int32[]
      end

      new_topo = get_grid_topology(new_model)
      new_cell_to_face_ids = Geometry.get_faces(new_topo,Dc,Df)

      return old_cell_to_face_entity, new_cell_to_face_ids
    end |> tuple_of_arrays

    # Redistribute entity data
    new_cell_to_face_entity = redistribute_cell_dofs(
      old_cell_to_face_entity,new_cell_to_face_ids,new_model,glue
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
  # The difficulty here is that String and Vector{Int32} are not isbits types.
  # We have to convert them to isbits types, then convert them back.
  name_data, name_ptrs, entities_data, entities_ptrs = map(old_models) do old_model
    if !isnothing(old_model)
      new_labels = get_face_labeling(old_model)
      names = JaggedArray(map(collect,new_labels.tag_to_name))
      entities = JaggedArray(new_labels.tag_to_entities)
      return names.data, names.ptrs, entities.data, entities.ptrs
    else
      return Char[], Int32[], Int32[], Int32[]
    end
  end |> tuple_of_arrays

  name_data = emit(name_data)
  entities_data = emit(entities_data)
  name_ptrs = emit(name_ptrs)
  entities_ptrs = emit(entities_ptrs)
  
  new_tag_to_name, new_tag_to_entities = map(
    name_data,name_ptrs,entities_data,entities_ptrs
  ) do name_data, name_ptrs, entities_data, entities_ptrs
    names = Vector{String}(undef,length(name_ptrs)-1)
    for i = 1:length(names)
      names[i] = join(Char.(name_data[name_ptrs[i]:name_ptrs[i+1]-1]))
    end
    entities = Vector{Vector{Int32}}(undef,length(entities_ptrs)-1)
    for i = 1:length(entities)
      entities[i] = entities_data[entities_ptrs[i]:entities_ptrs[i+1]-1]
    end
    return names, entities
  end |> tuple_of_arrays

  new_labels = map(FaceLabeling,new_d_to_dface_to_entity,new_tag_to_entities,new_tag_to_name)
  return DistributedFaceLabeling(new_labels)
end

# UnstructuredDiscreteModel refinement

function Adaptivity.refine(
  cmodel::DistributedUnstructuredDiscreteModel{Dc},args...;kwargs...
) where Dc
  fmodels, f_own_to_local = refine_local_models(cmodel,args...;kwargs...)
  fgids = refine_cell_gids(cmodel,fmodels,f_own_to_local)
  metadata = DistributedAdaptedDiscreteModelCache(
    nothing,cmodel.metadata,get_cell_gids(cmodel)
  )
  return GenericDistributedDiscreteModel(fmodels,fgids;metadata)
end

"""
    refine_local_models(cmodel::DistributedDiscreteModel{Dc},args...;kwargs...) where Dc

Given a coarse model, returns the locally refined models. This is done by 
  - refining the local models serially
  - filtering out the extra fine layers of ghosts
We also return the ids of the owned fine cells.

To find the fine cells we want to keep, we have the following criteria: 
  - Given a fine cell, it is owned iff 
    A) It's parent cell is owned
  - Given a fine cell, it is a ghost iff not(A) and 
    B) It has at least one neighbor with a non-ghost parent

Instead of checking A and B, we do the following: 
  - We mark fine owned cells by checking A 
  - If a cell is owned, we set it's fine neighbors as owned or ghost
"""
function refine_local_models(
  cmodel::DistributedDiscreteModel{Dc},args...;kwargs...
) where Dc
  cgids = partition(get_cell_gids(cmodel))
  cmodels = local_views(cmodel)

  # Refine models locally
  fmodels = map(cmodels) do cmodel
    refine(cmodel,args...;kwargs...)
  end

  # Select fine cells we want to keep
  Df = 0 # Dimension used to find neighboring cells
  f_own_or_ghost_ids, f_own_ids = map(cgids,cmodels,fmodels) do cgids,cmodel,fmodel
    glue = get_adaptivity_glue(fmodel)
    f2c_map = glue.n2o_faces_map[Dc+1]
    child_map = glue.n2o_cell_to_child_id

    ftopo = get_grid_topology(fmodel)
    f_cell_to_facet = Geometry.get_faces(ftopo,Dc,Df)
    f_facet_to_cell = Geometry.get_faces(ftopo,Df,Dc)
    f_cell_to_facet_cache = array_cache(f_cell_to_facet)
    f_facet_to_cell_cache = array_cache(f_facet_to_cell)
    c_l2o_map = local_to_own(cgids)
    
    f_own_mask = fill(false,length(f2c_map))
    f_own_or_ghost_mask = fill(false,length(f2c_map))
    for (fcell,ccell) in enumerate(f2c_map)
      if !iszero(c_l2o_map[ccell])
        f_own_mask[fcell] = true
        facets = getindex!(f_cell_to_facet_cache,f_cell_to_facet,fcell)
        for facet in facets
          facet_cells = getindex!(f_facet_to_cell_cache,f_facet_to_cell,facet)
          for facet_cell in facet_cells
            f_own_or_ghost_mask[facet_cell] = true
          end
        end
      end
    end

    f_own_or_ghost_ids = findall(f_own_or_ghost_mask)
    f_own_ids = findall(i -> f_own_mask[i],f_own_or_ghost_ids) # ModelPortion numeration

    return f_own_or_ghost_ids, f_own_ids
  end |> tuple_of_arrays

  # Filter out local models
  filtered_fmodels = map(fmodels,f_own_or_ghost_ids) do fmodel,f_own_or_ghost_ids
    model = UnstructuredDiscreteModel( # Necessary to keep the same type
      DiscreteModelPortion(get_model(fmodel),f_own_or_ghost_ids)
    )
    parent = get_parent(fmodel)

    _glue = get_adaptivity_glue(fmodel)
    n2o_faces_map = Vector{Vector{Int}}(undef,Dc+1)
    n2o_faces_map[Dc+1] = _glue.n2o_faces_map[Dc+1][f_own_or_ghost_ids]
    n2o_cell_to_child_id = _glue.n2o_cell_to_child_id[f_own_or_ghost_ids]
    rrules = _glue.refinement_rules
    glue = AdaptivityGlue(n2o_faces_map,n2o_cell_to_child_id,rrules)
    return AdaptedDiscreteModel(model,parent,glue)
  end

  return filtered_fmodels, f_own_ids
end

"""
    refine_cell_gids(
      cmodel::DistributedDiscreteModel{Dc},
      fmodels::AbstractArray{<:DiscreteModel{Dc}}
    ) where Dc

Given a coarse model and it's local refined models, returns the gids of the fine model.
The gids are computed as follows: 
  - First, we create a global numbering for the owned cells by adding an owner-based offset to the local 
    cell ids (such that cells belonging to the first processor are numbered first). This is 
    quite standard.
  - The complicated part is making this numeration consistent, i.e communicating gids of the 
    ghost cells. To do so, each processor selects it's ghost fine cells, and requests their 
    global ids by sending two keys:
      1. The global id of the coarse parent
      2. The child id of the fine cell
"""
function refine_cell_gids(
  cmodel::DistributedDiscreteModel{Dc},
  fmodels::AbstractArray{<:DiscreteModel{Dc}},
  f_own_to_local::AbstractArray{<:AbstractArray{Int}},
) where Dc

  cgids = partition(get_cell_gids(cmodel))
  cmodels = local_views(cmodel)
  ranks = linear_indices(cgids)

  # Create own numbering (without ghosts)
  num_f_owned_cells = map(length,f_own_to_local)
  num_f_gids = reduce(+,num_f_owned_cells)
  first_f_gid = scan(+,num_f_owned_cells,type=:exclusive,init=1)
  
  own_fgids = map(ranks,first_f_gid,num_f_owned_cells) do rank,first_f_gid,num_f_owned_cells
    f_o2g = collect(first_f_gid:first_f_gid+num_f_owned_cells-1)
    own   = OwnIndices(num_f_gids,rank,f_o2g)
    ghost = GhostIndices(num_f_gids) # No ghosts
    return OwnAndGhostIndices(own,ghost)
  end
  
  # Select ghost fine cells local ids
  parts_rcv, parts_snd = assembly_neighbors(cgids);
  lids_snd = map(parts_snd,cgids,fmodels) do parts_snd,cgids,fmodel
    glue = get_adaptivity_glue(fmodel)
    f2c_map = glue.n2o_faces_map[Dc+1]
    c_owners = local_to_owner(cgids)
    return JaggedArray(map(p -> findall(parent -> c_owners[parent] == p,f2c_map),parts_snd))
  end
  
  # Given the local ids of the fine cells we want to get info on, we 
  # collect two keys: 
  #   1. The global id of the coarse parent
  #   2. The child id of the fine cell
  parent_gids_snd, child_ids_snd = map(cgids,cmodels,fmodels,lids_snd) do cgids,cmodel,fmodel,lids_snd
    glue = get_adaptivity_glue(fmodel)
    f2c_map = glue.n2o_faces_map[Dc+1]
    child_map = glue.n2o_cell_to_child_id
  
    c_l2g_map = local_to_global(cgids)
    c_owners  = local_to_owner(cgids)
  
    parent_gids, child_ids, owners = map(lids_snd.data) do fcell
      ccell = f2c_map[fcell]
      return c_l2g_map[ccell], child_map[fcell], c_owners[ccell]
    end |> tuple_of_arrays
  
    ptrs = lids_snd.ptrs
    return JaggedArray(parent_gids,ptrs), JaggedArray(child_ids,ptrs)
  end |> tuple_of_arrays;
  
  # We exchange the keys
  parts_rcv, parts_snd = assembly_neighbors(cgids);
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t1 = exchange(parent_gids_snd,graph)
  t2 = exchange(child_ids_snd,graph)
  parent_gids_rcv = fetch(t1)
  child_ids_rcv = fetch(t2)
  
  # We process the received keys, and collect the global ids of the fine cells
  # that have been requested by our neighbors.
  child_gids_rcv = map(
    cgids,own_fgids,f_own_to_local,cmodels,fmodels,parent_gids_rcv,child_ids_rcv
  ) do cgids,own_fgids,f_own_to_local,cmodel,fmodel,parent_gids_rcv,child_ids_rcv
    glue = get_adaptivity_glue(fmodel)
    c2f_map = glue.o2n_faces_map
    child_map = glue.n2o_cell_to_child_id
    c2f_map_cache = array_cache(c2f_map)
  
    parent_lids = to_local!(parent_gids_rcv.data,cgids)
    child_ids = child_ids_rcv.data
  
    f_local_to_own = Arrays.find_inverse_index_map(f_own_to_local)
  
    child_lids = map(parent_lids,child_ids) do ccell, child_id
      fcells = getindex!(c2f_map_cache,c2f_map,ccell)
      pos = findfirst(fcell -> child_map[fcell] == child_id, fcells)
      return f_local_to_own[fcells[pos]]
    end
  
    child_gids = to_global!(child_lids,own_fgids)
    return JaggedArray(child_gids,parent_gids_rcv.ptrs)
  end
  
  # We exchange back the information
  graph = ExchangeGraph(parts_rcv,parts_snd)
  t = exchange(child_gids_rcv,graph)
  child_gids_snd = fetch(t)
  
  # We finally can create the global numeration of the fine cells by piecing together: 
  #   1. The (local ids,global ids) of the owned fine cells
  #   2. The (owners,local ids,global ids) of the ghost fine cells
  fgids = map(
    ranks,f_own_to_local,own_fgids,parts_snd,lids_snd,child_gids_snd
  ) do rank,own_lids,own_gids,nbors,ghost_lids,ghost_gids

    own2global = own_to_global(own_gids)
  
    n_own   = length(own_lids)
    n_ghost = length(ghost_lids.data)
    local2global = fill(0,n_own+n_ghost)
    local2owner = fill(0,n_own+n_ghost)
  
    # Own cells
    for (oid,lid) in enumerate(own_lids)
      local2global[lid] = own2global[oid]
      local2owner[lid]  = rank
    end
    
    # Ghost cells
    for (n,nbor) in enumerate(nbors)
      for i in ghost_lids.ptrs[n]:ghost_lids.ptrs[n+1]-1
        lid = ghost_lids.data[i]
        gid = ghost_gids.data[i]
        local2global[lid] = gid
        local2owner[lid]  = nbor
      end
    end
    return LocalIndices(num_f_gids,rank,local2global,local2owner)
  end

  return PRange(fgids)
end

function refine_cell_gids(
  cmodel::DistributedDiscreteModel{Dc},
  fmodels::AbstractArray{<:DiscreteModel{Dc}}
) where Dc
  cgids = partition(get_cell_gids(cmodel))
  f_own_to_local = map(cgids,fmodels) do cgids,fmodel
    glue = get_adaptivity_glue(fmodel)
    f2c_map = glue.n2o_faces_map[Dc+1]
    @assert isa(f2c_map,Vector) "Only uniform refinement is supported!"
  
    c_l2o_map = local_to_own(cgids)
    return findall(parent -> !iszero(c_l2o_map[parent]),f2c_map)
  end
  return refine_cell_gids(cmodel,fmodels,f_own_to_local)
end
