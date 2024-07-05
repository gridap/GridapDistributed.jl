
using PartitionedArrays, GridapDistributed
using Gridap
using Gridap.Adaptivity, Gridap.Arrays, Gridap.Geometry

using GridapDistributed: GenericDistributedDiscreteModel

ranks = with_debug() do distribute
  distribute(LinearIndices((2,)))
end

Dc = 2

_cmodel = CartesianDiscreteModel(ranks,(2,1),(0,1,0,1),(4,4))
cmodel = GenericDistributedDiscreteModel(
  map(UnstructuredDiscreteModel,local_views(_cmodel)),
  get_cell_gids(_cmodel),
)

# A) Refine local models
cgids = partition(get_cell_gids(cmodel))
cmodels = local_views(cmodel)
fmodels = map(cmodels) do cmodel
  refine(cmodel)
end

# B) Remove extra ghost cells

f_ghost_lids = map(cgids,cmodels,fmodels) do cgids,cmodel,fmodel
  glue = get_adaptivity_glue(fmodel)
  f2c_map = glue.n2o_faces_map[Dc+1]
  child_map = glue.n2o_cell_to_child_id

  ftopo = get_grid_topology(fmodel)
  f_cell_to_cell = Geometry.get_faces(ftopo,Dc,Dc) # Not what we want, we need to go through faces
  f_cell_to_cell_cache = array_cache(f_cell_to_cell)
  c_l2o_map = local_to_own(cgids)
  
  f_ghost_mask = fill(false,length(f2c_map))
  for (fcell,ccell) in enumerate(f2c_map)
    fine_nbors = getindex!(f_cell_to_cell_cache,f_cell_to_cell,fcell)
    println(fine_nbors)
    A = iszero(c_l2o_map[ccell]) # Parent is ghost
    B = any(nbor -> !iszero(c_l2o_map[f2c_map[nbor]]), fine_nbors) # Has neighbor with non-ghost parent
    f_ghost_mask[fcell] = A && B
  end

  return findall(f_ghost_mask)
end


# C) Create global numeration of own cells by
#   1. Counting the number of owned fine cells in each model (children of owned coarse cells)
#   2. Computing the first global id of each model, and the maximum gobal id
#   3. Creating a global numeration of owned fine cells
f_own_to_local = map(cgids,cmodels,fmodels) do cgids,cmodel,fmodel
  glue = get_adaptivity_glue(fmodel)
  f2c_map = glue.n2o_faces_map[Dc+1]
  child_map = glue.n2o_cell_to_child_id
  @assert isa(f2c_map,Vector) "Only uniform refinement is supported!"

  c_l2o_map = local_to_own(cgids)
  return findall(parent -> !iszero(c_l2o_map[parent]),f2c_map)
end

num_f_owned_cells = map(length,f_own_to_local)
num_f_gids = reduce(+,num_f_owned_cells)
first_f_gid = scan(+,num_f_owned_cells,type=:exclusive,init=1)

own_fgids = map(ranks,first_f_gid,num_f_owned_cells) do rank,first_f_gid,num_f_owned_cells
  f_o2g = collect(first_f_gid:first_f_gid+num_f_owned_cells-1)
  own   = OwnIndices(num_f_gids,rank,f_o2g)
  ghost = GhostIndices(num_f_gids) # No ghosts
  return OwnAndGhostIndices(own,ghost)
end

# D) Exchange ghost gids by sending two keys: 
#   1. The global id of the coarse parent
#   2. The child id of the fine cell

# This has extra ghosts, but will do for now (see above)
parts_rcv, parts_snd = assembly_neighbors(cgids);
lids_snd = map(ranks,parts_snd,cgids,cmodels,fmodels) do r,parts_snd,cgids,cmodel,fmodel
  glue = get_adaptivity_glue(fmodel)
  f2c_map = glue.n2o_faces_map[Dc+1]
  child_map = glue.n2o_cell_to_child_id
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
fgids = map(ranks,f_own_to_local,own_fgids,lids_snd,child_gids_snd) do rank, own_lids, own_gids, ghost_lids, ghost_gids
  own2global = own_to_global(own_gids)

  n_nbors = length(ghost_lids)
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
  for nbor in 1:n_nbors
    for i in ghost_lids.ptrs[nbor]:ghost_lids.ptrs[nbor+1]-1
      lid = ghost_lids.data[i]
      gid = ghost_gids.data[i]
      local2global[lid] = gid
      local2owner[lid]  = nbor
    end
  end
  return LocalIndices(num_f_gids,rank,local2global,local2owner)
end

