
using Gridap
using Gridap.Geometry
using Gridap.Adaptivity

using GridapDistributed
using PartitionedArrays

function DistributedAdaptivityGlue(serial_glue,parent,child)
  glue = map(partition(get_cell_gids(parent)),partition(get_cell_gids(child))) do parent_gids, child_gids
    old_l2g = parent_gids.local_to_global
    new_l2g = child_gids.local_to_global
  
    n2o_faces_map = [Int64[],Int64[],serial_glue.n2o_faces_map[3][new_l2g]]
    n2o_cell_to_child_id = serial_glue.n2o_cell_to_child_id[new_l2g]
    rrules = serial_glue.refinement_rules[old_l2g]
    AdaptivityGlue(n2o_faces_map,n2o_cell_to_child_id,rrules)
  end
  return glue
end

############################################################################################

coarse_ranks = with_debug() do distribute
  distribute(LinearIndices((2,)))
end

fine_ranks = with_debug() do distribute
  distribute(LinearIndices((4,)))
end

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

redist_child_cell_to_part = lazy_map(Reindex(redist_parent_cell_to_part),serial_rglue.n2o_faces_map[3])
redist_child = DiscreteModel(fine_ranks,serial_child,redist_child_cell_to_part)
fine_adaptivity_glue = DistributedAdaptivityGlue(serial_rglue,redist_parent,redist_child)

# Building the RedistributeGlue

old_cell_to_part = parent_cell_to_part
new_cell_to_part = redist_parent_cell_to_part

p = 1

old_partition = partition(get_cell_gids(parent)).items[p]
new_partition = partition(get_cell_gids(redist_parent)).items[p]

# This looks wrong... why is it like this? 
global_to_local(old_partition)

old_new_cell_to_part = collect(zip(old_cell_to_part,new_cell_to_part))
gids_rcv = findall(x->x[1]!=p && x[2]==p, old_new_cell_to_part)
gids_snd = findall(x->x[1]==p && x[2]!=p, old_new_cell_to_part)

parts_rcv = unique(old_cell_to_part[gids_rcv])
parts_snd = unique(new_cell_to_part[gids_snd])
lids_rcv = [findall(x -> x == nbor ,old_cell_to_part[gids_rcv]) for nbor in parts_rcv]
lids_snd = [findall(x -> x == nbor ,new_cell_to_part[gids_snd]) for nbor in parts_snd]


map(local_views(redist_child)) do model
  coords  = Geometry.get_node_coordinates(model)
  c2n_map = Geometry.get_faces(get_grid_topology(model),2,0)
  cell_coords = lazy_map(nodes->lazy_map(Reindex(coords),nodes),c2n_map)
  display(cell_coords)
  return nothing
end

