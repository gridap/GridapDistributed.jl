
# DistributedAdaptedDiscreteModels

const DistributedAdaptedDiscreteModel{Dc,Dp} = GenericDistributedDiscreteModel{Dc,Dp,<:AbstractArray{<:AdaptedDiscreteModel{Dc,Dp}}}

function DistributedAdaptedDiscreteModel(model  ::DistributedDiscreteModel,
                                         parent ::DistributedDiscreteModel,
                                         glue   ::AbstractArray{<:AdaptivityGlue})
  models = map(local_views(model),local_views(parent),glue) do model, parent, glue
    AdaptedDiscreteModel(model,parent,glue)
  end
  return GenericDistributedDiscreteModel(models,get_cell_gids(model))
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

function get_redistribute_cell_dofs_cache(cell_dof_values_old,
                                          cell_dof_ids_new,
                                          model_new,
                                          glue::RedistributeGlue;
                                          reverse=false)

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

function redistribute_cell_dofs(cell_dof_values_old,
                                cell_dof_ids_new,
                                model_new,
                                glue::RedistributeGlue;
                                reverse=false)
  caches = get_redistribute_cell_dofs_cache(cell_dof_values_old,cell_dof_ids_new,model_new,glue;reverse=reverse)
  return redistribute_cell_dofs!(caches,cell_dof_values_old,cell_dof_ids_new,model_new,glue;reverse=reverse)
end

function redistribute_cell_dofs!(caches,
                                 cell_dof_values_old,
                                 cell_dof_ids_new,
                                 model_new,
                                 glue::RedistributeGlue;
                                 reverse=false)

  snd_data, rcv_data, cell_dof_values_new = caches
  lids_rcv, lids_snd, parts_rcv, parts_snd, new2old = get_glue_components(glue,Val(reverse))
  old_parts, new_parts = get_old_and_new_parts(glue,Val(reverse))

  cell_dof_values_old = change_parts(cell_dof_values_old,get_parts(glue);default=[])
  cell_dof_ids_new    = change_parts(cell_dof_ids_new,get_parts(glue);default=[[]])

  _pack_snd_data!(snd_data,cell_dof_values_old,lids_snd)

  graph = ExchangeGraph(parts_snd,parts_rcv)
  t = exchange!(rcv_data,snd_data,graph)
  wait(t)

  # We have to build the owned part of "cell_dof_values_new" out of
  #  1. cell_dof_values_old (for those cells s.t. new2old[:]!=0)
  #  2. cell_dof_values_new_rcv (for those cells s.t. new2old[:]=0)
  _update_cell_dof_values_with_local_info!(cell_dof_values_new,
                                           cell_dof_values_old,
                                           new2old)

  _unpack_rcv_data!(cell_dof_values_new,rcv_data,lids_rcv)

  # Now that every part knows it's new owned dofs, exchange ghosts
  cell_dof_values_new = change_parts(cell_dof_values_new,new_parts)
  if i_am_in(new_parts)
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

function get_redistribute_free_values_cache(fv_new::Union{PVector,Nothing},
                                            Uh_new::Union{DistributedSingleFieldFESpace,Nothing},
                                            fv_old::Union{PVector,Nothing},
                                            dv_old::Union{AbstractArray,Nothing},
                                            Uh_old::Union{DistributedSingleFieldFESpace,Nothing},
                                            model_new,
                                            glue::RedistributeGlue;
                                            reverse=false)
  old_parts, new_parts = get_old_and_new_parts(glue,Val(reverse))
  cell_dof_values_old = i_am_in(old_parts) ? map(scatter_free_and_dirichlet_values,local_views(Uh_old),local_views(fv_old),dv_old) : nothing
  cell_dof_ids_new    = i_am_in(new_parts) ? map(_get_cell_dof_ids_inner_space, local_views(Uh_new)) : nothing
  caches = get_redistribute_cell_dofs_cache(cell_dof_values_old,cell_dof_ids_new,model_new,glue;reverse=reverse)
  return caches
end

function redistribute_free_values(fv_new::Union{PVector,Nothing},
                                  Uh_new::Union{DistributedSingleFieldFESpace,Nothing},
                                  fv_old::Union{PVector,Nothing},
                                  dv_old::Union{AbstractArray,Nothing},
                                  Uh_old::Union{DistributedSingleFieldFESpace,Nothing},
                                  model_new,
                                  glue::RedistributeGlue;
                                  reverse=false)

  caches = get_redistribute_free_values_cache(fv_new,Uh_new,fv_old,dv_old,Uh_old,model_new,glue;reverse=reverse)
  return redistribute_free_values!(caches,fv_new,Uh_new,fv_old,dv_old,Uh_old,model_new,glue;reverse=reverse)
end

function redistribute_free_values!(caches,
                                   fv_new::Union{PVector,Nothing},
                                   Uh_new::Union{DistributedSingleFieldFESpace,Nothing},
                                   fv_old::Union{PVector,Nothing},
                                   dv_old::Union{AbstractArray,Nothing},
                                   Uh_old::Union{DistributedSingleFieldFESpace,Nothing},
                                   model_new,
                                   glue::RedistributeGlue;
                                   reverse=false)

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

function redistribute_fe_function(uh_old::Union{DistributedSingleFieldFEFunction,Nothing},
                                  Uh_new::Union{DistributedSingleFieldFESpace,Nothing},
                                  model_new,
                                  glue::RedistributeGlue;
                                  reverse=false)

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
