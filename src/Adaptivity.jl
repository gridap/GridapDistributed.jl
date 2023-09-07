
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

get_parts(g::RedistributeGlue) = biggest_parts(g.new_parts,g.old_parts)

function PartitionedArrays.reverse(g::RedistributeGlue)
  return RedistributeGlue(g.old_parts,g.new_parts,g.parts_snd,g.parts_rcv,
                          g.lids_snd,g.lids_rcv,g.new2old,g.old2new)
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

function _copy_local_info!(values_new,values_old,new2old)
  map(values_new,values_old,new2old) do values_new,values_old,new2old
    ocache = array_cache(values_old)
    for (ncell,ocell) in enumerate(new2old)
      if ocell!=0
        # Copy old cell to new cell
        oentry = getindex!(ocache,values_old,ocell)
        range  = values_new.ptrs[ncell]:values_new.ptrs[ncell+1]-1
        values_new.data[range] .= oentry
      end
    end
  end
end

function _allocate_comm_data(cell_values,lids;T=Float64)
  map(cell_values,lids) do cell_values,lids
    cache = array_cache(cell_values)
    num_nbors = length(lids)
    ptrs = fill(zero(Int32),num_nbors+1)
    for nbor = 1:num_nbors
      for j = lids.ptrs[nbor]:lids.ptrs[nbor+1]-1
        cell = lids.data[j]
        ptrs[nbor+1] += length(getindex!(cache,cell_values,cell))
      end
    end
    PartitionedArrays.length_to_ptrs!(ptrs)

    data = Vector{T}(undef,ptrs[end]-1)
    PartitionedArrays.JaggedArray(data,ptrs)
  end
end

function _pack_snd_data!(snd_data,cell_values,snd_lids)
  map(snd_data,cell_values,snd_lids) do snd_data,cell_values,snd_lids
    cache = array_cache(cell_values)
    s = 1
    for i = 1:length(snd_lids)
      for j = snd_lids.ptrs[i]:snd_lids.ptrs[i+1]-1
        cell  = snd_lids.data[j]
        ldofs = getindex!(cache,cell_values,cell)

        e = s+length(ldofs)-1
        range = s:e
        snd_data.data[range] .= ldofs
        s = e+1
      end
    end
  end
end

function _unpack_rcv_data!(cell_values,rcv_data,rcv_lids)
  map(cell_values,rcv_data,rcv_lids) do cell_values,rcv_data,rcv_lids
    s = 1
    for i = 1:length(rcv_lids.ptrs)-1
      for j = rcv_lids.ptrs[i]:rcv_lids.ptrs[i+1]-1
        cell = rcv_lids.data[j]
        range_cell_values = cell_values.ptrs[cell]:cell_values.ptrs[cell+1]-1
        
        e = s+length(range_cell_values)-1
        range_rcv_data = s:e
        cell_values.data[range_cell_values] .= rcv_data.data[range_rcv_data]
        s = e+1
      end
    end
  end
end

function get_redistribute_cache(values_old,
                                values_new,
                                glue::RedistributeGlue)
  exchange_parts = biggest_parts(glue.new_parts,glue.old_parts)
  values_old = change_parts(values_old,exchange_parts;default=[])
  values_new = change_parts(values_new,exchange_parts;default=[[]])

  snd_data = _allocate_comm_data(values_old, glue.lids_snd)
  rcv_data = _allocate_comm_data(values_new, glue.lids_rcv)

  caches = snd_data, rcv_data
  return caches
end

function redistribute(a_old::PVector,
                      a_new::PVector,
                      glue::RedistributeGlue)
  values_old = partition(a_old)
  values_new = partition(a_new)
  redistribute_cache = get_redistribute_cache(values_old,values_new,glue)
  exchange_cache_new = a_new.cache
  return redistribute!(redistribute_cache,values_old,values_new,exchange_cache_new,glue)
end

function redistribute(values_old,
                      values_new,
                      indices_new,
                      glue::RedistributeGlue)
  redistribute_cache = get_redistribute_cache(values_old,values_new,glue)
  exchange_cache_new = fetch_vector_ghost_values_cache(values_new,indices_new)
  return redistribute!(redistribute_cache,values_old,values_new,exchange_cache_new,glue)
end

function redistribute!(caches,
                       values_old,
                       values_new,
                       exchange_cache_new,
                       glue::RedistributeGlue)

  snd_data, rcv_data = caches

  exchange_parts = biggest_parts(glue.new_parts,glue.old_parts)
  values_old = change_parts(values_old,exchange_parts;default=[])

  _pack_snd_data!(snd_data,values_old,glue.lids_snd)

  graph = ExchangeGraph(glue.parts_snd,glue.parts_rcv)
  t = exchange!(rcv_data,snd_data,graph)

  # We have to build the owned part of "cell_dof_values_new" out of
  #  1. cell_values_old (for those cells s.t. new2old[:]!=0)
  #  2. cell_values_new_rcv (for those cells s.t. new2old[:]=0)
  _copy_local_info!(values_new,values_old,glue.new2old)

  wait(t)
  _unpack_rcv_data!(values_new,rcv_data,glue.lids_rcv)

  # Now that every part knows it's new owned dofs, exchange ghosts
  values_new = change_parts(values_new,glue.new_parts)
  if i_am_in(glue.new_parts)
    fetch_vector_ghost_values!(values_new,exchange_cache_new) |> wait
  end
  return values_new
end

function biggest_parts(p1,p2)
  if length(p1) > length(p2)
    return p1
  else
    return p2
  end
end
biggest_parts(p1::Nothing,p2) = p2
biggest_parts(p1,p2::Nothing) = p1

"""
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
                                            glue::RedistributeGlue)
  old_parts, new_parts = get_old_and_new_parts(glue)
  cell_dof_values_old = i_am_in(old_parts) ? map(scatter_free_and_dirichlet_values,local_views(Uh_old),local_views(fv_old),dv_old) : nothing
  cell_dof_ids_new    = i_am_in(new_parts) ? map(_get_cell_dof_ids_inner_space, local_views(Uh_new)) : nothing
  caches = get_redistribute_cell_dofs_cache(cell_dof_values_old,cell_dof_ids_new,model_new,glue)
  return caches
end

function redistribute_free_values(fv_new::Union{PVector,Nothing},
                                  Uh_new::Union{DistributedSingleFieldFESpace,Nothing},
                                  fv_old::Union{PVector,Nothing},
                                  dv_old::Union{AbstractArray,Nothing},
                                  Uh_old::Union{DistributedSingleFieldFESpace,Nothing},
                                  model_new,
                                  glue::RedistributeGlue)

  caches = get_redistribute_free_values_cache(fv_new,Uh_new,fv_old,dv_old,Uh_old,model_new,glue)
  return redistribute_free_values!(caches,fv_new,Uh_new,fv_old,dv_old,Uh_old,model_new,glue)
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
"""