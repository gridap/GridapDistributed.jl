
using Gridap
using Gridap.Geometry
using Gridap.Adaptivity
using Gridap.FESpaces

using GridapDistributed
using PartitionedArrays

using GridapDistributed: find_local_to_local_map
using GridapDistributed: RedistributeGlue, redistribute_cell_dofs, redistribute_fe_function, redistribute_free_values

function DistributedAdaptivityGlue(serial_glue,parent,child)
  glue = map(partition(get_cell_gids(parent)),partition(get_cell_gids(child))) do parent_gids, child_gids
    old_l2g = local_to_global(parent_gids)
    new_l2g = local_to_global(child_gids)
  
    n2o_faces_map = [Int64[],Int64[],serial_glue.n2o_faces_map[3][new_l2g]]
    n2o_cell_to_child_id = serial_glue.n2o_cell_to_child_id[new_l2g]
    rrules = serial_glue.refinement_rules[old_l2g]
    AdaptivityGlue(n2o_faces_map,n2o_cell_to_child_id,rrules)
  end
  return glue
end

function get_redistribute_glue(old_parts::T,new_parts::T,old_cell_to_part,new_cell_to_part,model,redist_model) where {T<:PartitionedArrays.DebugArray}
  parts_rcv,parts_snd,lids_rcv,lids_snd,old2new,new2old = 
  map(new_parts,partition(get_cell_gids(redist_model))) do p,new_partition
    old_new_cell_to_part = collect(zip(old_cell_to_part,new_cell_to_part))
    gids_rcv = findall(x->x[1]!=p && x[2]==p, old_new_cell_to_part)
    gids_snd = findall(x->x[1]==p && x[2]!=p, old_new_cell_to_part)

    parts_rcv = unique(old_cell_to_part[gids_rcv])
    parts_snd = unique(new_cell_to_part[gids_snd])

    gids_rcv_by_part = [filter(x -> old_cell_to_part[x] == nbor, gids_rcv) for nbor in parts_rcv]
    gids_snd_by_part = [filter(x -> new_cell_to_part[x] == nbor, gids_snd) for nbor in parts_snd]

    if p ∈ old_parts.items
      old_partition = partition(get_cell_gids(model)).items[p]
      lids_rcv = map(gids -> lazy_map(Reindex(global_to_local(new_partition)),gids),gids_rcv_by_part)
      lids_snd = map(gids -> lazy_map(Reindex(global_to_local(old_partition)),gids),gids_snd_by_part)
      old2new = replace(find_local_to_local_map(old_partition,new_partition), -1 => 0)
      new2old = replace(find_local_to_local_map(new_partition,old_partition), -1 => 0)
    else
      lids_rcv = map(gids -> lazy_map(Reindex(global_to_local(new_partition)),gids),gids_rcv_by_part)
      lids_snd = map(gids -> fill(Int32(0),length(gids)),gids_snd_by_part)
      old2new = Int[]
      new2old = fill(0,length(findall(x -> x == p,new_cell_to_part)))
    end

    return parts_rcv,parts_snd,JaggedArray(lids_rcv),JaggedArray(lids_snd),old2new,new2old
  end |> tuple_of_arrays

  return RedistributeGlue(parts_rcv,parts_snd,lids_rcv,lids_snd,old2new,new2old)
end

############################################################################################

coarse_ranks = with_debug() do distribute
  distribute(LinearIndices((2,)))
end

fine_ranks = with_debug() do distribute
  distribute(LinearIndices((4,)))
end

# Create models and glues 

serial_parent = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(4,4)))
serial_child  = refine(serial_parent)
serial_rglue  = get_adaptivity_glue(serial_child)

parent_cell_to_part = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]
parent = DiscreteModel(coarse_ranks,serial_parent,parent_cell_to_part)

child_cell_to_part = lazy_map(Reindex(parent_cell_to_part),serial_rglue.n2o_faces_map[3])
child = DiscreteModel(coarse_ranks,serial_child,child_cell_to_part)
coarse_adaptivity_glue = DistributedAdaptivityGlue(serial_rglue,parent,child)

redist_parent_cell_to_part = [1,1,2,2,1,1,2,2,3,3,4,4,3,3,4,4]
redist_parent = DiscreteModel(fine_ranks,serial_parent,redist_parent_cell_to_part)
redist_glue_parent = get_redistribute_glue(coarse_ranks,fine_ranks,parent_cell_to_part,redist_parent_cell_to_part,parent,redist_parent);

redist_child_cell_to_part = lazy_map(Reindex(redist_parent_cell_to_part),serial_rglue.n2o_faces_map[3])
redist_child = DiscreteModel(fine_ranks,serial_child,redist_child_cell_to_part)
fine_adaptivity_glue = DistributedAdaptivityGlue(serial_rglue,redist_parent,redist_child)
redist_glue_child = get_redistribute_glue(coarse_ranks,fine_ranks,child_cell_to_part,redist_child_cell_to_part,child,redist_child);

# Create FESpaces

reffe = ReferenceFE(lagrangian,Float64,1)
parent_space = FESpace(parent,reffe)
child_space  = FESpace(child,reffe)
redist_parent_space = FESpace(redist_parent,reffe)
redist_child_space  = FESpace(redist_child,reffe)

sol(x) = sum(x)
parent_u = interpolate(sol,parent_space)
child_u  = interpolate(sol,child_space)
redist_parent_u = interpolate(sol,redist_parent_space)
redist_child_u  = interpolate(sol,redist_child_space)

parent_cell_dofs = map(get_cell_dof_values,local_views(parent_u))
child_cell_dofs  = map(get_cell_dof_values,local_views(child_u))
redist_parent_cell_dofs = map(get_cell_dof_values,local_views(redist_parent_u))
redist_child_cell_dofs  = map(get_cell_dof_values,local_views(redist_child_u))

parent_free_values = map(get_free_dof_values,local_views(parent_u))
child_free_values  = map(get_free_dof_values,local_views(child_u))
redist_parent_free_values = map(get_free_dof_values,local_views(redist_parent_u))
redist_child_free_values  = map(get_free_dof_values,local_views(redist_child_u))

tmp_cell_dofs = copy(redist_parent_cell_dofs)
redistribute_cell_dofs(parent_cell_dofs,tmp_cell_dofs,redist_parent,redist_glue_parent)

