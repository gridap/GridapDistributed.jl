using Test

using Gridap
using Gridap.Geometry
using Gridap.Adaptivity
using Gridap.FESpaces
using Gridap.Arrays

using MPI
using GridapDistributed
using PartitionedArrays

using GridapDistributed: i_am_in, generate_subparts
using GridapDistributed: find_local_to_local_map
using GridapDistributed: DistributedAdaptedDiscreteModel
using GridapDistributed: RedistributeGlue, redistribute

function are_equal(a1::Union{MPIArray,DebugArray},a2::Union{MPIArray,DebugArray})
  same = map(a1,a2) do a1,a2
    a1 ≈ a2
  end
  return reduce(&,same,init=true)
end

function are_equal(a1::PVector,a2::PVector)
  are_equal(own_values(a1),own_values(a2))
end

function get_redistribute_glue(old_parts,new_parts::DebugArray,old_cell_to_part,new_cell_to_part,model,redist_model)
  parts_rcv,parts_snd,lids_rcv,lids_snd,old2new,new2old = 
  map(new_parts,partition(get_cell_gids(redist_model))) do p,new_partition
    old_new_cell_to_part = collect(zip(old_cell_to_part,new_cell_to_part))
    gids_rcv = findall(x->x[1]!=p && x[2]==p, old_new_cell_to_part)
    gids_snd = findall(x->x[1]==p && x[2]!=p, old_new_cell_to_part)

    parts_rcv = unique(old_cell_to_part[gids_rcv])
    parts_snd = unique(new_cell_to_part[gids_snd])

    lids_rcv = [filter(x -> old_cell_to_part[x] == nbor, gids_rcv) for nbor in parts_rcv]
    map(gids -> to_local!(gids,new_partition),lids_rcv)

    if p ∈ old_parts.items
      old_partition = partition(get_cell_gids(model)).items[p]
      lids_snd = [filter(x -> new_cell_to_part[x] == nbor, gids_snd) for nbor in parts_snd]
      map(gids -> to_local!(gids,old_partition),lids_snd)
      old2new  = replace(find_local_to_local_map(old_partition,new_partition), -1 => 0)
      new2old  = replace(find_local_to_local_map(new_partition,old_partition), -1 => 0)
    else
      lids_snd = [Int[]]
      old2new  = Int[]
      new2old  = fill(0,length(findall(x -> x == p,new_cell_to_part)))
    end

    return parts_rcv,parts_snd,JaggedArray(lids_rcv),JaggedArray(lids_snd),old2new,new2old
  end |> tuple_of_arrays

  return RedistributeGlue(new_parts,old_parts,parts_rcv,parts_snd,lids_rcv,lids_snd,old2new,new2old)
end

function fetch_ghost_info(old_cell_lids,new_cell_lids,new_dof_indices,glue)
  glue_parts = get_parts(glue)

  _old_cell_lids = GridapDistributed.change_parts(old_cell_lids,glue_parts;default=[Int32[]])
  _new_cell_lids = GridapDistributed.change_parts(new_cell_lids,glue_parts;default=[Int32[]])
  ghost_rcv_data = GridapDistributed._allocate_comm_data(_old_cell_lids,glue.lids_snd;T=Bool)
  ghost_snd_data = GridapDistributed._allocate_comm_data(_new_cell_lids,glue.lids_rcv;T=Bool)

  cellwise_isghost = map(new_cell_lids,new_dof_indices) do cell_lids, dof_indices
    lids_to_isghost = local_to_own(dof_indices) .== 0
    data = lazy_map(Reindex(lids_to_isghost),cell_lids.data)
    Table(data,cell_lids.ptrs)
  end

  GridapDistributed._pack_snd_data!(ghost_snd_data,cellwise_isghost,glue.lids_rcv)
  graph = ExchangeGraph(glue.parts_rcv,glue.parts_snd)
  t = exchange!(ghost_rcv_data,ghost_snd_data,graph)
  return t
end

function get_dof_lids_rcv(glue,indices,cell_lids)
  glue_parts = get_parts(glue)
  parts  = linear_indices(indices)
  dof_lids_rcv = map(glue_parts,glue.lids_snd) do p, lids_snd
    if p ∈ parts.items
      _indices   = indices.items[p]
      _cell_lids = cell_lids.items[p]
      lids_to_own = local_to_own(_indices)
  
      touched = Vector{Bool}(undef,local_length(_indices))
      ptrs = fill(0,length(lids_snd)+1)
      for nbor = 1:length(lids_snd)
        fill!(touched,false)
        for cell = lids_snd.ptrs[nbor]:lids_snd.ptrs[nbor+1]-1
          cell_lid = lids_snd.data[cell]
          for l in _cell_lids.ptrs[cell_lid]:_cell_lids.ptrs[cell_lid+1]-1
            dof_lid = _cell_lids.data[l]
            dof_isghost = (lids_to_own[dof_lid] == 0)
            if !touched[dof_lid] && !dof_isghost
              ptrs[nbor+1] += 1
              touched[dof_lid] = true
            end
          end
        end
      end
      PartitionedArrays.length_to_ptrs!(ptrs)
  
      data = Vector{Int}(undef,ptrs[end]-1)
      for nbor = 1:length(lids_snd)
        s = 0
        fill!(touched,false)
        for cell = lids_snd.ptrs[nbor]:lids_snd.ptrs[nbor+1]-1
          cell_lid = lids_snd.data[cell]
          for l in _cell_lids.ptrs[cell_lid]:_cell_lids.ptrs[cell_lid+1]-1
            dof_lid = _cell_lids.data[l]
            dof_isghost = (lids_to_own[dof_lid] == 0)
            if !touched[dof_lid] && !dof_isghost
              data[ptrs[nbor]+s] = dof_lid
              touched[dof_lid] = true
              s += 1
            end
          end
        end
      end
      return PartitionedArrays.JaggedArray(data,ptrs)
    else
      return PartitionedArrays.JaggedArray(Int[],Int[])
    end
  end
  return dof_lids_rcv
end

function get_dof_lids_snd(glue,indices,cell_lids,is_ghost)
  glue_parts = get_parts(glue)
  parts = linear_indices(indices)
  dof_lids_snd = map(glue_parts,glue.lids_snd,is_ghost) do p, lids_snd, is_ghost
    if p ∈ parts.items
      _indices   = indices.items[p]
      _cell_lids = cell_lids.items[p]
  
      touched = Vector{Bool}(undef,local_length(_indices))
      ptrs = fill(0,length(lids_snd)+1)
      for nbor = 1:length(lids_snd)
        dof = 0
        fill!(touched,false)
        for cell in lids_snd.ptrs[nbor]:lids_snd.ptrs[nbor+1]-1
          cell_lid = lids_snd.data[cell]
          for l in _cell_lids.ptrs[cell_lid]:_cell_lids.ptrs[cell_lid+1]-1
            dof_lid = _cell_lids.data[l]
            dof_isghost = is_ghost.data[is_ghost.ptrs[nbor]+dof]
            if !touched[dof_lid] && !dof_isghost
              ptrs[nbor+1] += 1
              touched[dof_lid] = true
            end
            dof += 1
          end
        end
      end
      PartitionedArrays.length_to_ptrs!(ptrs)
  
      data = Vector{Int}(undef,ptrs[end]-1)
      for nbor = 1:length(lids_snd)
        s = 0
        dof = 0
        fill!(touched,false)
        for cell = lids_snd.ptrs[nbor]:lids_snd.ptrs[nbor+1]-1
          cell_lid = lids_snd.data[cell]
          for l in _cell_lids.ptrs[cell_lid]:_cell_lids.ptrs[cell_lid+1]-1
            dof_lid = _cell_lids.data[l]
            dof_isghost = is_ghost.data[is_ghost.ptrs[nbor]+dof]
            if !touched[dof_lid] && !dof_isghost
              data[ptrs[nbor]+s] = dof_lid
              touched[dof_lid] = true
              s += 1
            end
            dof += 1
          end
        end
      end
      return PartitionedArrays.JaggedArray(data,ptrs)
    else
      return PartitionedArrays.JaggedArray(Int[],Int[])
    end
  end
  return dof_lids_snd
end

function get_old_to_new(
  glue,
  old_cell_indices,new_cell_indices,
  old_dof_indices,new_dof_indices,
  old_cell_lids,new_cell_lids)

  old_parts  = linear_indices(old_dof_indices)
  new_parts  = linear_indices(new_dof_indices)
  glue_parts = get_parts(glue)
  dof_old_to_new = map(glue_parts,glue.old2new) do p, cells_old_to_new
    if (p <= length(old_parts)) && (p <= length(new_parts))
      dof_old_to_new = fill(0,local_length(old_dof_indices.items[p]))
      old_cell_lids_part = old_cell_lids.items[p]
      new_cell_lids_part = new_cell_lids.items[p]
      # For each old cell that is also in the new partition
      for (old_cell_lid,new_cell_lid) in enumerate(cells_old_to_new)
        if (new_cell_lid != 0)
          # For each dof in the old cell, find new dof lid
          num_dofs = old_cell_lids_part.ptrs[old_cell_lid+1]-old_cell_lids_part.ptrs[old_cell_lid]
          for dof in 0:num_dofs-1
            old_dof_lid = old_cell_lids_part.data[old_cell_lids_part.ptrs[old_cell_lid]+dof]
            new_dof_lid = new_cell_lids_part.data[new_cell_lids_part.ptrs[new_cell_lid]+dof]
            dof_old_to_new[old_dof_lid] = new_dof_lid # Save mapping
          end
        end
      end
    elseif p <= length(old_parts)
      dof_old_to_new = fill(0,local_length(old_dof_indices.items[p]))
    else
      dof_old_to_new = Int[]
    end
    return dof_old_to_new
  end
  return dof_old_to_new
end

############################################################################################

fine_ranks = with_debug() do distribute
  distribute(LinearIndices((4,)))
end
coarse_ranks = generate_subparts(fine_ranks,2)

# Create models and glues 
serial_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(4,4)))
model_cell_to_part = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]
model = DiscreteModel(coarse_ranks,serial_model,model_cell_to_part)

redist_model_cell_to_part = [1,1,2,2,1,1,2,2,3,3,4,4,3,3,4,4]
redist_model = DiscreteModel(fine_ranks,serial_model,redist_model_cell_to_part)
redist_glue = get_redistribute_glue(coarse_ranks,fine_ranks,model_cell_to_part,redist_model_cell_to_part,model,redist_model);

# FESpaces and functions
sol(x) = sum(x)
reffe  = ReferenceFE(lagrangian,Float64,1)

space = FESpace(model,reffe)
u = interpolate(sol,space)
cell_dofs = map(get_cell_dof_values,local_views(u))
free_values = get_free_dof_values(u)
dir_values = zero_dirichlet_values(space)

redist_space = FESpace(redist_model,reffe)
redist_u = interpolate(sol,redist_space)
redist_cell_dofs = map(get_cell_dof_values,local_views(redist_u))
redist_free_values = get_free_dof_values(redist_u)
redist_dir_values = zero_dirichlet_values(redist_space)

# Redistribute cell values, both ways
redist_indices = partition(get_cell_gids(redist_model))
tmp_cell_dofs = map(PartitionedArrays.JaggedArray,copy(redist_cell_dofs))
redistribute(cell_dofs,tmp_cell_dofs,redist_indices,redist_glue)
@test are_equal(redist_cell_dofs,tmp_cell_dofs)

indices = partition(get_cell_gids(model))
tmp_cell_dofs = map(PartitionedArrays.JaggedArray,copy(cell_dofs))
redistribute(redist_cell_dofs,tmp_cell_dofs,indices,reverse(redist_glue))
@test are_equal(cell_dofs,tmp_cell_dofs)

# Creating RedistributeGlue for dofs

glue = redist_glue;
old_dof_indices = partition(space.gids)
new_dof_indices = partition(redist_space.gids)

old_cell_indices = partition(get_cell_gids(model))
new_cell_indices = partition(get_cell_gids(redist_model))

old_cell_lids = map(get_cell_dof_ids,local_views(space))
new_cell_lids = map(get_cell_dof_ids,local_views(redist_space))

glue_parts = get_parts(glue)
new_parts = linear_indices(new_cell_indices)
old_parts = linear_indices(old_cell_indices)

dof_old_to_new = get_old_to_new(glue,old_cell_indices,new_cell_indices,old_dof_indices,new_dof_indices,old_cell_lids,new_cell_lids)
dof_new_to_old = get_old_to_new(reverse(glue),new_cell_indices,old_cell_indices,new_dof_indices,old_dof_indices,new_cell_lids,old_cell_lids)

_dof_new_to_old = map((o2n,idx) -> Arrays.find_inverse_index_map(o2n,local_length(idx)),dof_old_to_new,new_dof_indices)

t = fetch_ghost_info(old_cell_lids,new_cell_lids,new_dof_indices,glue)
new_cell_isghost = fetch(t)

t = fetch_ghost_info(new_cell_lids,old_cell_lids,old_dof_indices,reverse(glue))
old_cell_isghost = fetch(t)

#

dof_lids_snd = get_dof_lids_snd(glue,old_dof_indices,old_cell_lids,new_cell_isghost)
dof_lids_rcv = get_dof_lids_rcv(reverse(glue),new_dof_indices,new_cell_lids)

_dof_lids_snd = get_dof_lids_snd(reverse(glue),new_dof_indices,new_cell_lids,old_cell_isghost)
_dof_lids_rcv = get_dof_lids_rcv(glue,old_dof_indices,old_cell_lids)

"""
  snd is correct, but rcv is not
  Reason: The parent model cannot deduce info on ghost dofs on the child model. We need to 
  cummunicate this info from the child to the parent. 
  Solution: We allocate and communicate a JaggedArray of Bools with the info and communicate 
  from new to old, so that the old model can correctly determine what is ghost and what is not...
"""



# Redistribute free values, both ways
tmp_free_values = copy(redist_free_values)
redistribute_free_values(tmp_free_values,redist_space,free_values,dir_values,space,redist_model,redist_glue)
@test are_equal(redist_free_values,tmp_free_values)

tmp_free_values = i_am_in(coarse_ranks) ? copy(free_values) : nothing
redistribute_free_values(tmp_free_values,space,redist_free_values,redist_dir_values,redist_space,model,redist_glue;reverse=true)
if i_am_in(coarse_ranks)
  @test are_equal(free_values,tmp_free_values)
end



