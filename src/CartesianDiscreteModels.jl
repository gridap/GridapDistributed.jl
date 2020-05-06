function Gridap.CartesianDiscreteModel(comm::Communicator,subdomains::Tuple,args...)
  desc = CartesianDescriptor(args...)
  CartesianDiscreteModel(comm,subdomains,desc)
end

function Gridap.CartesianDiscreteModel(
  comm::Communicator,subdomains::Tuple,gdesc::CartesianDescriptor)

  nsubdoms = prod(subdomains)
  ngcells = prod(Tuple(gdesc.partition))

  models = DistributedData(comm) do isubdom

    ldesc = local_cartesian_descriptor(gdesc,subdomains,isubdom)
    CartesianDiscreteModel(ldesc)
  end

  gids = DistributedIndexSet(comm,ngcells) do isubdom

    lid_to_gid, lid_to_owner = local_cartesian_gids(gdesc,subdomains,isubdom)
    IndexSet(ngcells,lid_to_gid,lid_to_owner)
  end

  do_on_parts(comm,models,gids) do part,model,gid
      adjust_local_labels_to_reflect_global_facelabeling!(model,gid,gdesc.partition)
  end

  DistributedDiscreteModel(models,gids)
end

function local_cartesian_descriptor_1d(
  gdesc::CartesianDescriptor{1},nsubdoms::Integer,isubdom::Integer)

  gcells, = gdesc.partition
  gorigin, = gdesc.origin
  h, = gdesc.sizes
  H = h*gcells/nsubdoms

  orange = uniform_partition_1d(gcells,nsubdoms,isubdom)
  ocells = length(orange)

  if nsubdoms == 1
    lcells =  ocells
    lorigin = gorigin
  elseif isubdom == 1
    lcells =  ocells + 1
    lorigin = gorigin
  elseif isubdom != nsubdoms
    lcells = ocells + 2
    lorigin = gorigin + (first(orange)-2)*h
  else
    lcells = ocells + 1
    lorigin = gorigin + (first(orange)-2)*h
  end

  CartesianDescriptor(lorigin,h,lcells,gdesc.map)

end

function local_cartesian_gids_1d(
  gdesc::CartesianDescriptor{1},nsubdoms::Integer,isubdom::Integer)

  gcells, = gdesc.partition

  orange = uniform_partition_1d(gcells,nsubdoms,isubdom)
  ocells = length(orange)

  if nsubdoms == 1
    lrange = orange
  elseif isubdom == 1
    lrange = orange.start:(orange.stop+1)
  elseif isubdom != nsubdoms
    lrange = (orange.start-1):(orange.stop+1)
  else
    lrange = (orange.start-1):orange.stop
  end

  lcells = length(lrange)
  lid_to_gid = collect(Int,lrange)
  lid_to_owner = fill(isubdom,lcells)

  if nsubdoms == 1
    nothing
  elseif isubdom == 1
    lid_to_owner[end] = 2
  elseif isubdom != nsubdoms
    lid_to_owner[1] = isubdom - 1
    lid_to_owner[end] = isubdom + 1
  else
    lid_to_owner[1] = isubdom - 1
  end

  lid_to_gid, lid_to_owner
end

function local_cartesian_descriptor(gdesc::CartesianDescriptor,nsubdoms::Tuple,isubdom::Integer)
  cis = CartesianIndices(nsubdoms)
  ci = cis[isubdom]
  local_cartesian_descriptor(gdesc,nsubdoms,ci)
end

function local_cartesian_descriptor(
  gdesc::CartesianDescriptor{D,T},nsubdoms::Tuple,isubdom::CartesianIndex) where {D,T}

  origin = zeros(T,D)
  sizes = zeros(T,D)
  partition = zeros(Int,D)
  for d in 1:D
    gdesc_d = CartesianDescriptor(gdesc.origin[d],gdesc.sizes[d],gdesc.partition[d])
    ldesc_d = local_cartesian_descriptor_1d(gdesc_d,nsubdoms[d],isubdom[d])
    origin[d] = ldesc_d.origin[1]
    sizes[d] = ldesc_d.sizes[1]
    partition[d] = ldesc_d.partition[1]
  end

  CartesianDescriptor(origin,sizes,partition,gdesc.map)
end

function local_cartesian_gids(
  gdesc::CartesianDescriptor{D},nsubdoms::Tuple,isubdom::Integer) where D
  cis = CartesianIndices(nsubdoms)
  ci = cis[isubdom]
  local_cartesian_gids(gdesc,nsubdoms,ci)
end

function local_cartesian_gids(
  gdesc::CartesianDescriptor{D},nsubdoms::Tuple,isubdom::CartesianIndex) where D

  d_to_lid_to_gid = Vector{Int}[]
  d_to_lid_to_owner = Vector{Int}[]
  for d in 1:D
    gdesc_d = CartesianDescriptor(gdesc.origin[d],gdesc.sizes[d],gdesc.partition[d])
    lid_to_gid_d, lid_to_owner_d = local_cartesian_gids_1d(gdesc_d,nsubdoms[d],isubdom[d])
    push!(d_to_lid_to_gid,lid_to_gid_d)
    push!(d_to_lid_to_owner,lid_to_owner_d)
  end

  d_to_llength = Tuple(map(length,d_to_lid_to_gid))
  d_to_glength = Tuple(gdesc.partition)

  lcis = CartesianIndices(d_to_llength)
  gcis = CartesianIndices(d_to_glength)
  scis = CartesianIndices(nsubdoms)
  llis = LinearIndices(lcis)
  glis = LinearIndices(gcis)
  slis = LinearIndices(scis)

  lid_to_gid = zeros(Int,length(lcis))
  lid_to_owner = zeros(Int,length(lcis))
  gci = zeros(Int,D)
  sci = zeros(Int,D)

  for lci in lcis
    for d in 1:D
      gci[d] = d_to_lid_to_gid[d][lci[d]]
      sci[d] = d_to_lid_to_owner[d][lci[d]]
    end
    lid = llis[lci]
    lid_to_gid[lid] = glis[CartesianIndex(Tuple(gci))]
    lid_to_owner[lid] = slis[CartesianIndex(Tuple(sci))]
  end

  lid_to_gid, lid_to_owner
end

function uniform_partition_1d(glength,np,pid)
  _olength = glength ÷ np
  _offset = _olength * (pid-1)
  _rem = glength % np
  if _rem < (np-pid+1)
    olength = _olength
    offset = _offset
  else
    olength = _olength + 1
    offset = _offset + pid - (np-_rem) - 1
  end
  (1+offset):(olength+offset)
end

function adjust_local_labels_to_reflect_global_facelabeling!(
  model::CartesianDiscreteModel{D},
  gid,
  gpartition,
) where {D}

  gcis = CartesianIndices(Tuple(gpartition))
  glis = LinearIndices(gcis)
  topo = model.grid_topology
  polytope = first(get_polytopes(topo))
  face_labeling = model.face_labeling
  offsets = Gridap.ReferenceFEs.get_offsets(polytope)

  polytope_d_face_to_jfaces =
    Matrix{Vector{Vector{Int}}}(undef, (D,D))
  for d = 0:(D-1)
    for j = d+1:D-1
      polytope_d_face_to_jfaces[d+1, j+1] =
        Gridap.ReferenceFEs.get_faces(polytope, d, j)
    end
  end

  # NOTE: The next loop is a generalized version of its (serial)
  #       counterpart in Gridap.Geometry.
  # TODO: Reduce code replication? Any strategy towards
  #       code replication implies modifications into Gridap.
  interior_id = Gridap.ReferenceFEs.num_faces(polytope)
  boundary_id = -1
  face_deltas = find_face_neighbours_deltas(model)
  for d = 0:(D-1)
    face_to_cells = Gridap.ReferenceFEs.get_faces(topo, d, D)
    cell_to_faces = Gridap.ReferenceFEs.get_faces(topo, D, d)
    face_to_geolabel = face_labeling.d_to_dface_to_entity[d+1]
    nfaces = length(face_to_geolabel)
    for face_gid = 1:nfaces
      # Restrict traversal to those d-faces which
      # are on the subdomain boundary
      if (face_to_geolabel[face_gid] != interior_id)
        cell_gid = face_to_cells.data[face_to_cells.ptrs[face_gid]]
        a = cell_to_faces.ptrs[cell_gid]
        b = cell_to_faces.ptrs[cell_gid+1] - 1

        face_lid = -1
        for j = a:b
          if (cell_to_faces.data[j] == face_gid)
            face_lid = j - a + 1
            break
          end
        end
        @assert face_lid != -1

        face_lid += offsets[d+1]
        # Check whether cell neighbour across face face_lid belongs to the
        # global grid. If yes, the current face is actually at the interior
        is_assigned_face_delta = isassigned(face_deltas, face_lid)
        if (
          is_assigned_face_delta &&
          (gcis[gid.lid_to_gid[cell_gid]] + face_deltas[face_lid]) in gcis
        )
          face_to_geolabel[face_gid] = interior_id
        elseif (face_to_geolabel[face_gid] != face_lid)
          face_to_geolabel[face_gid] = boundary_id
        else
          # If the entity label of the current face reflects that of the global grid,
          # then there must be only one cell around in the global mesh as well
          cell_found = false
          for j = d+1:D-1
            dface_to_jfaces =
              polytope_d_face_to_jfaces[d+1, j+1][face_lid-offsets[d+1]]
            cell_found = _is_there_interior_cell_across_higher_dim_faces(
              dface_to_jfaces,
              offsets[j+1],
              gcis,
              gid.lid_to_gid[cell_gid],
              face_deltas,
            )
            cell_found && break
          end
          if (cell_found)
            face_to_geolabel[face_gid] = boundary_id
          end
        end
      end
    end
  end

  # NOTE: The following nested loop was copied "as-is" from Gridap.geometry
  # TODO: Reduce code replication. Any strategy towards
  #       code replication implies modifications into Gridap.
  for d = 0:(D-2)
    for j = (d+1):(D-1)
      dface_to_jfaces = Gridap.ReferenceFEs.get_faces(topo, d, j)
      dface_to_geolabel = face_labeling.d_to_dface_to_entity[d+1]
      jface_to_geolabel = face_labeling.d_to_dface_to_entity[j+1]
      Gridap.Geometry._fix_dface_geolabels!(
        dface_to_geolabel,
        jface_to_geolabel,
        dface_to_jfaces.data,
        dface_to_jfaces.ptrs,
        interior_id,
        boundary_id,
      )
    end
  end
end

function _is_there_interior_cell_across_higher_dim_faces(
  dface_to_jfaces,
  offset_j,
  gcis,
  cell_gid,
  face_deltas,
)
  for k in dface_to_jfaces
    jface_lid = k + offset_j
    if (isassigned(face_deltas, jface_lid))
      gci = gcis[cell_gid]
      if ((gci + face_deltas[jface_lid]) in gcis)
        return true
      end
    end
  end
  return false
end

"""
  _find_face_neighbour(model::CartesianDiscreteModel{D}, cell_gid, face_lid) where {D}
    -> Union{Nothing,get_data_eltype(eltype(model.grid_topology.n_m_to_nface_to_mfaces))}

  Given a D-dimensional CartesianDiscreteModel{D}, a cell K with global ID cell_gid,
  and a d-face F, d < D, with local identifier face_lid within K (i.e., global ID
  face_lid within Polytope{P} associated to the model, i.e., SEGMENT, QUAD, HEX, etc.),
  returns the global ID of the (unique) neighbour of K across F (if it exists) or nothing
  (if it doesn't)

  NOTE/TODO: I am positive that the performance of the current version of the subroutine
  could be much improved if Polytope{D} provides a helper function that given a face_lid
  returns what has to be added to the CartesianIndex of a cell in order to obtain the
  CartesianIndex of the cell neighbour of K across F. This would be a very lightweight
  function that exploits the numbering scheme of the cells and faces within cells for a
  Cartesian-like mesh as coded in n-cube type Polytopes.

"""
function _find_face_neighbour(
  model::CartesianDiscreteModel{D},
  cell_gid,
  face_lid,
) where {D}
  p = first(model.grid_topology.polytopes)
  num_faces = Gridap.ReferenceFEs.num_faces(p)
  offsets = Gridap.ReferenceFEs.get_offsets(p)
  topo = model.grid_topology
  @assert 1 <= face_lid <= num_faces
  d = Gridap.ReferenceFEs.get_facedims(p)[face_lid] # Dimension of the face
  @assert 0 <= d < D
  cells_to_dfaces = Gridap.ReferenceFEs.get_faces(topo, D, d)
  dfaces_to_cells = Gridap.ReferenceFEs.get_faces(topo, d, D)
  face_gid = cells_to_dfaces[cell_gid][face_lid-offsets[d+1]]
  dface_to_cells = dfaces_to_cells[face_gid]
  dface_to_cells = Set(dface_to_cells)
  setdiff!(dface_to_cells, [cell_gid])
  for j = d+1:D-1
    dface_to_jfaces =
      Set(Gridap.ReferenceFEs.get_faces(p, d, j)[face_lid-offsets[d+1]])
    for jface_lid in dface_to_jfaces
      jfaces_to_cells = Gridap.ReferenceFEs.get_faces(topo, j, D)
      cells_to_jfaces = Gridap.ReferenceFEs.get_faces(topo, D, j)
      jface_gid = cells_to_jfaces[cell_gid][jface_lid]
      jface_to_cells = jfaces_to_cells[jface_gid]
      setdiff!(dface_to_cells, jface_to_cells)
    end
  end
  @assert length(dface_to_cells) in [0, 1]
  result = nothing
  for element in dface_to_cells
    result = element
  end
  result
end

"""
  _find_face_neighbours_deltas(model :: CartesianDiscreteModel) -> Vector{CartesianIndex}

  Given a CartesianDiscreteModel, returns V=Vector{CartesianIndex} with as many
  entries as the number of faces in the boundary of the Polytope associated to
  model. For an entry face_lid in this vector, V[face_lid] returns what has to be added to the
  CartesianIndex of a cell in order to obtain the CartesianIndex of the cell neighbour of K
  across the face F with local ID face_lid.
"""
function find_face_neighbours_deltas(model::CartesianDiscreteModel{D}) where {D}
  desc = Gridap.Geometry.get_cartesian_descriptor(model.grid)
  cis = CartesianIndices(Tuple(desc.partition))
  lis = LinearIndices(cis)
  p = first(model.grid_topology.polytopes)
  num_faces = Gridap.ReferenceFEs.num_faces(p)
  delta_faces = Vector{CartesianIndex}(undef, num_faces - 1)
  num_completed_faces = 0
  completed_faces = fill(false, num_faces - 1)

  ci_corners = NTuple{D,Int}[]
  cartesian_indices_of_corner_cells!(ci_corners, (), Tuple(desc.partition))

  for ci_corner in ci_corners
    cell_gid = lis[ci_corner...]
    for face_lid = 1:num_faces-1
      if (!completed_faces[face_lid])
        neighbour = _find_face_neighbour(model, cell_gid, face_lid)
        if (neighbour != nothing)
          ci_cell_gid = cis[cell_gid]
          ci_cell_neighbour = cis[neighbour]
          # delta is defined s.t. ci_cell_neighbour = ci_cell_gid+delta
          delta = ci_cell_neighbour - ci_cell_gid
          delta_faces[face_lid] = delta
          completed_faces[face_lid] = true
          num_completed_faces += 1
          num_completed_faces == num_faces - 1 && break
        end
      end
    end
    num_completed_faces == num_faces - 1 && break
  end
  delta_faces
end

#TODO: Is there any other (existing) alternative of doing this using the tools
#provided by the Gridap.Geometry namespace?
function cartesian_indices_of_corner_cells!(
  corners::AbstractVector{NTuple{L,Int}},
  index::NTuple{K,Int},
  partition::NTuple{N,Int},
) where {L,K,N}
  @assert L == K + N
  if (N == 0)
    return push!(corners, index)
  end
  for i in (1, partition[1])
    cartesian_indices_of_corner_cells!(corners, (index..., i), partition[2:end])
  end
  corners
end
