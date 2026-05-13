
"""
    struct MacroDiscreteModel{Dc,Dp} <: DistributedDiscreteModel{Dc,Dp}

DistributedModel of the interfaces between the processors. 

Each d-interface is given as an agglomeration of d-faces, and a global numbering 
is produced for the interfaces. 

## Constructors: 

    MacroDiscreteModel(model::DistributedDiscreteModel)

## Visualizing the interfaces:

Interfaces have two different numberings. One is local to each processor, and the 
other is global and consistent across all processors. To extract and visualize 
the interfaces, you can use the following functions:

- `get_local_face_labeling(macro_model::MacroDiscreteModel{Dc})`
- `get_global_face_labeling(macro_model::MacroDiscreteModel{Dc})`
- `writevtk_local(macro_model::MacroDiscreteModel{Dc},filename::String;vtk_kwargs...)`
- `writevtk_global(macro_model::MacroDiscreteModel{Dc},filename::String;vtk_kwargs...)`

"""
struct MacroDiscreteModel{Dc,Dp,A,B,C} <: DistributedDiscreteModel{Dc,Dp}
  model::A
  d_to_interfaces::B
  interface_gids::C
  function MacroDiscreteModel(
    model::DistributedDiscreteModel{Dc,Dp},
    d_to_interfaces::AbstractArray{<:JaggedArray{<:JaggedArray}},
    interface_gids::PRange
  ) where {Dc,Dp}
    A = typeof(model)
    B = typeof(d_to_interfaces)
    C = typeof(interface_gids)
    return new{Dc,Dp,A,B,C}(model,d_to_interfaces,interface_gids)
  end
end

function MacroDiscreteModel(model::DistributedDiscreteModel)
  d_to_interfaces = classify_interfaces(model;sort_faces=true)
  nbors, keys     = generate_nbors_and_keys(model,d_to_interfaces;is_sorted=true)
  interface_gids  = generate_interface_gids(nbors,keys)
  return MacroDiscreteModel(model,d_to_interfaces,interface_gids)
end

# DistributedDiscreteModel API

function get_cell_gids(macro_model::MacroDiscreteModel)
  return linear_indices(local_views(macro_model.model))
end

# TODO: Reuse macro_model.interface_gids
function get_face_gids(macro_model::MacroDiscreteModel,dim::Integer)
  if isequal(dim,num_cell_dims(macro_model))
    return get_cell_gids(macro_model)
  end

  model = macro_model.model
  d_to_interfaces = macro_model.d_to_interfaces
  nbors, keys = generate_nbors_and_keys(model,d_to_interfaces;is_sorted=true,dimensions=[dim])
  return generate_interface_gids(nbors,keys)
end

function get_local_dimranges(macro_model::MacroDiscreteModel{Dc}) where Dc
  dranges = map(macro_model.d_to_interfaces) do d_to_interfaces
    ptrs = [1,map(length,d_to_interfaces)...]
    PartitionedArrays.length_to_ptrs!(ptrs)
    return map(d -> ptrs[d]:ptrs[d+1]-1,1:Dc)
  end
  return dranges
end

"""
  Returns a JaggedArray of JaggedArrays such that we have

    [interface dimension][interface lid][face dimension] -> interface dfaces lids
  
  I.e we bundle interfaces of the same dimension together (macro-faces, macro-edges, macro-nodes), 
  and for each interface we return a JaggedArray with the d-faces in the interface.

  If `sort_faces` is true, the faces in each interface are sorted by gid. If false, the faces
  are by default sorted by lid.
"""
function classify_interfaces(model::DistributedDiscreteModel{Dc};sort_faces=true) where Dc
  ranks = linear_indices(local_views(model))
  topo  = get_grid_topology(model)
  d_to_fgids = map(Df -> partition(get_face_gids(model,Df)),0:Dc) |> to_parray_of_arrays
  
  # Classify dfaces into interfaces: d_to_interface_to_d_to_dfaces
  d_to_interfaces = map(ranks,local_views(topo),d_to_fgids) do rank, topo, d_to_fgids
    cgids = d_to_fgids[Dc+1]
    cell_owners = local_to_owner(cgids)

    # Count dfaces in each interface
    owners_to_interface = Dict{UInt64,Int8}()
    ptrs = Vector{Int32}[]
    num_interfaces = 0
    d_to_num_interfaces = zeros(Int8,Dc+1)
    for Df = Dc-1:-1:0
      f2c_map = Geometry.get_faces(topo,Df,Dc)
      f2c_map_cache = array_cache(f2c_map)
      for face in 1:num_faces(topo,Df)
        cells  = getindex!(f2c_map_cache,f2c_map,face)
        owners = Set(view(cell_owners,cells))
        if face_is_interface(owners,rank)
          key = hash(owners)
          if haskey(owners_to_interface,key)
            interface = owners_to_interface[key]
            ptrs[interface][Df+2] += 1 
          else
            num_interfaces += 1
            d_to_num_interfaces[Df+2] += 1
            owners_to_interface[hash(owners)] = num_interfaces
            push!(ptrs,zeros(Int32,Dc+1)); ptrs[num_interfaces][Df+2] = 1
          end
        end
      end
    end
    map(PartitionedArrays.length_to_ptrs!,ptrs)
    PartitionedArrays.length_to_ptrs!(d_to_num_interfaces)

    # Collect interface faces
    data = map(p -> Vector{Int32}(undef,p[end]-1), ptrs)
    for Df = Dc-1:-1:0
      # Collect dfaces in each interface
      f2c_map = Geometry.get_faces(topo,Df,Dc)
      f2c_map_cache = array_cache(f2c_map)
      for face in 1:num_faces(topo,Df)
        cells  = getindex!(f2c_map_cache,f2c_map,face)
        owners = Set(view(cell_owners,cells))
        if face_is_interface(owners,rank)
          key = hash(owners)
          interface = owners_to_interface[key]
          data[interface][ptrs[interface][Df+1]] = face
          ptrs[interface][Df+1] += 1
        end
      end
    end
    map(PartitionedArrays.rewind_ptrs!,ptrs)

    interfaces = reverse(map(JaggedArray,data,ptrs))
    d_to_interfaces = JaggedArray(interfaces,d_to_num_interfaces)
    return d_to_interfaces
  end

  # Sort face gids in each interface
  if sort_faces
    map(d_to_interfaces,d_to_fgids) do d_to_interfaces, d_to_fgids
      interfaces = d_to_interfaces.data
      d_to_faces_l2g = map(local_to_global,d_to_fgids)
      for interface in interfaces
        data, ptrs = interface.data, interface.ptrs
        for Df in 0:Dc-1
          faces_l2g   = d_to_faces_l2g[Df+1]
          to_gid(lid) = faces_l2g[lid]
          sort!(view(data,ptrs[Df+1]:ptrs[Df+2]-1),by=to_gid)
        end
      end
    end
  end

  return d_to_interfaces
end

"""
    function generate_interface_gids(nbors,keys) -> gids

  Generates a global numbering for the interfaces, given two input arrays per processor:

    - `nbors`: For each interface, the minimum rank of the neighboring processors.
    - `keys`: For each interface, the minimum gid of the faces in the interface.
  
  Then the gids are assigned in the following way: 
  We iterate over the processors in ascending order. For each processor, we iterate over the local
  interfaces. Then: 

    - If the `nbor` processor of an interface has a higher (or equal) rank than the current processor, it means we
    haven't assigned a gid to that interface yet. So we assign a new gid to the current interface.
    - If the `nbor` processor of an interface has a lower rank than the current processor, it means we 
    already assigned a gid to that interface (while iterating over `nbor`). So we look for a matching `key`
    in the `keys` array of the `nbor` processor, and assign the same gid to the current interface.
"""
function generate_interface_gids(nbors,keys)
  # Gather to main
  _nbors = gather(nbors)
  _keys  = gather(keys)

  # Generate gids
  _igids, _n_glob = map(_nbors,_keys) do nbors, keys
    if !i_am_main(nbors)
      return nothing, nothing
    end

    nprocs = length(nbors)
    gid_data = Vector{Int64}(undef,length(keys.data))
  
    gid = 0
    for proc in 1:nprocs
      range = keys.ptrs[proc]:keys.ptrs[proc+1]-1
      for k in range
        nbor = nbors.data[k]
        key  = keys.data[k]
        if nbor >= proc # Interface not yet seen -> Assign gid
          gid += 1
          gid_data[k] = gid
        else            # Interface has been seen by processor `nbor` -> Look for gid within its gids
          gid_range = keys.ptrs[nbor]:keys.ptrs[nbor+1]-1
          pos = findfirst(i -> (keys.data[i]==key),gid_range)
          gid_data[k] = gid_data[keys.ptrs[nbor]+pos-1]
        end
      end
    end
  
    JaggedArray(gid_data,keys.ptrs), gid
  end |> tuple_of_arrays

  # Scatter to processors
  igids = scatter(_igids)
  n_glob = emit(_n_glob)

  # Create PRange
  ranks = linear_indices(igids)
  indices = map(ranks,n_glob,igids,nbors) do rank,n_glob, igids, nbors
    LocalIndices(n_glob,rank,igids,nbors)
  end
  return PRange(indices)
end

"""
  Given a model and a set of local interfaces for each model, returns
    - `nbors`: For each interface, the minimum rank of the neighboring processors.
    - `keys`: For each interface, the minimum gid of the faces in the interface.

  Options: 
   - `is_sorted`: If true, the keys are assumed to be sorted by gid. This allows some optimization.
   - `dimensions`: List/Set of interface dimensions to be considered. 
"""
function generate_nbors_and_keys(
  model::DistributedDiscreteModel{Dc},
  d_to_interfaces;
  is_sorted = false,
  dimensions = 0:Dc-1
) where Dc

  n_interfaces = map(d_to_interfaces) do d_to_interfaces
    sum(map(d -> length(d_to_interfaces[d+1]),dimensions))
  end

  # A) Get minimum rank of neighboring processors for each interface
  topo  = get_grid_topology(model)
  cgids = get_cell_gids(model)
  nbors = map(n_interfaces,d_to_interfaces,local_views(topo),partition(cgids)) do n_interfaces, d_to_interfaces, topo, cgids
    cell_owners = local_to_owner(cgids)
    to_owner(lid) = cell_owners[lid]

    # For each interface, we consider any face and take the minimum rank of the neighboring processors. 
    # All faces should have the same neighboring processors.
    i = 1
    nbors = Vector{Int64}(undef,n_interfaces)
    for (d,interfaces) in enumerate(d_to_interfaces)
      Df = d-1
      if Df ∈ dimensions
        f2c_map = Geometry.get_faces(topo,Df,Dc)
        f2c_map_cache = array_cache(f2c_map)
        for d_to_faces in interfaces
          faces = d_to_faces[Df+1]
          cells = getindex!(f2c_map_cache,f2c_map,first(faces))
          nbors[i] = minimum(to_owner,cells)
          i += 1
        end
      end
    end
    return nbors
  end

  # B) Get minimum gid of faces in each interface
  d_to_fgids = map(Df -> partition(get_face_gids(model,Df)),0:Dc-1) |> to_parray_of_arrays
  offsets = [0,map(Df -> num_faces(model,Df),0:Dc-1)...]
  PartitionedArrays.length_to_ptrs!(offsets)
  keys = map(n_interfaces,d_to_interfaces,d_to_fgids) do n_interfaces, d_to_interfaces, d_to_fgids
    i = 1
    keys = Vector{Int64}(undef,n_interfaces)
    for (d,interfaces) in enumerate(d_to_interfaces)
      Df = d-1
      if Df ∈ dimensions
        lid_to_gid   = local_to_global(d_to_fgids[Df+1])
        to_gid(lid)  = lid_to_gid[lid]
        for d_to_faces in interfaces
          faces = d_to_faces[Df+1]
          keys[i] = (is_sorted) ? to_gid(first(faces)) : minimum(to_gid,faces)
          keys[i] += offsets[Df+1] # This is to distinguish between interfaces of different dimensions
          i += 1
        end
      end
    end
    return keys
  end

  return nbors, keys
end

"""
  Returns a global (consistent) face labeling for the macro model. Requires communication.

  The face labeling contains the following tags: 
    - `Interior_i` : Faces which are interior to processor `i`.
    - `Interface_j`: Faces which belong to interface `j` (global id).
    - `Interiors`  : Union of all `Interior_i` tags.
    - `Interfaces` : Union of all `Interface_j` tags.
"""
function get_global_face_labeling(macro_model::MacroDiscreteModel{Dc}) where Dc
  model = macro_model.model
  ranks = linear_indices(local_views(model))
  d_to_fgids = map(Df -> partition(get_face_gids(model,Df)),0:Dc) |> to_parray_of_arrays
  d_to_interfaces = macro_model.d_to_interfaces
  interface_gids = partition(macro_model.interface_gids)

  labels = map(d_to_fgids,d_to_interfaces, interface_gids) do d_to_fgids, d_to_interfaces, interface_gids
    tag_to_name = Vector{String}()
    tag_to_entities = Vector{Vector{Int32}}()
    d_to_dface_to_entity = [Vector{Int32}(undef,local_length(d_to_fgids[Df+1])) for Df in 0:Dc]

    n_ranks = length(ranks)
    n_interfaces = global_length(interface_gids)

    # Add tags for the interiors for each proc
    for rank in 1:n_ranks
      push!(tag_to_name,"Interior_$(rank)")
      push!(tag_to_entities,[rank])
    end
    for Df in Dc:-1:0
      face_to_owner = local_to_owner(d_to_fgids[Df+1])
      d_to_dface_to_entity[Df+1] .= face_to_owner
    end

    # Add tags for the interfaces
    lid_to_gid = local_to_global(interface_gids)
    for gid in 1:global_length(interface_gids)
      entity = gid + n_ranks
      push!(tag_to_name,"Interface_$(gid)"); push!(tag_to_entities,[entity])
    end
    for (lid,d_to_faces) in enumerate(d_to_interfaces.data)
      gid = lid_to_gid[lid]
      entity = gid + n_ranks
      for Df in 0:Dc-1
        faces = d_to_faces[Df+1]
        for face in faces
          d_to_dface_to_entity[Df+1][face] = entity
        end
      end
    end

    # Add tags for the unions
    push!(tag_to_name,"Interiors"); push!(tag_to_entities,collect(1:n_ranks))
    push!(tag_to_name,"Interfaces"); push!(tag_to_entities,collect(n_ranks+1:n_ranks+n_interfaces))

    return FaceLabeling(d_to_dface_to_entity,tag_to_entities,tag_to_name)
  end

  # Make face entities consistent
  tasks = []
  for Df in 0:Dc
    face_entities = map(l -> l.d_to_dface_to_entity[Df+1],labels)
    face_gids = get_face_gids(model,Df)
    f2e = PVector(face_entities,partition(face_gids))
    push!(tasks,consistent!(f2e))
  end
  map(wait,tasks)

  return DistributedFaceLabeling(labels)
end

"""
  Returns a local face labeling for the macro model. Does not require communication.
  WARNING: This is NOT consistent, it is sub-assembled. The same face might have 
           different labels in different processors.

  The face labeling contains the following tags: 
    - `Interior`   : Faces which are interior to the processor.
    - `Exterior`   : Faces which are exterior to the processor.
    - `Interface_i`: Faces which belong to interface `i` (local id).
    - `Interfaces` : Union of all `Interface_i` tags.
"""
function get_local_face_labeling(macro_model::MacroDiscreteModel{Dc}) where Dc
  model = macro_model.model
  ranks = linear_indices(local_views(model))
  d_to_fgids = map(Df -> partition(get_face_gids(model,Df)),0:Dc) |> to_parray_of_arrays
  d_to_interfaces = macro_model.d_to_interfaces

  labels = map(ranks,d_to_fgids,d_to_interfaces) do rank, d_to_fgids, d_to_interfaces
    tag_to_name = Vector{String}()
    tag_to_entities = Vector{Vector{Int32}}()
    d_to_dface_to_entity = [Vector{Int32}(undef,local_length(d_to_fgids[Df+1])) for Df in 0:Dc]

    # First we fill with owner data
    push!(tag_to_name,"Interior"); push!(tag_to_entities,[1])
    push!(tag_to_name,"Exterior"); push!(tag_to_entities,[2])
    for Df in Dc:-1:0
      face_to_owner = local_to_owner(d_to_fgids[Df+1])
      for (face,owner) in enumerate(face_to_owner)
        entity = (owner == rank) ? 1 : 2
        d_to_dface_to_entity[Df+1][face] = entity
      end
    end

    # Then we overwrite with interface data
    push!(tag_to_name,"Interfaces"); push!(tag_to_entities,[])
    for Df in 0:Dc-1
      push!(tag_to_name,"$(Df)_Interfaces"); push!(tag_to_entities,[])
    end

    interface_lid = 0
    for d in 0:Dc-1
      interfaces = d_to_interfaces[d+1]
      for d_to_faces in interfaces
        interface_lid += 1
        entity = interface_lid + 2
        push!(tag_to_name,"Interface_$(interface_lid)"); push!(tag_to_entities,[entity])
        push!(tag_to_entities[3],entity)
        push!(tag_to_entities[4+d],entity)
        for Df in 0:Dc-1
          faces = d_to_faces[Df+1]
          for face in faces
            d_to_dface_to_entity[Df+1][face] = entity
          end
        end
      end
    end

    return FaceLabeling(d_to_dface_to_entity,tag_to_entities,tag_to_name)
  end
  return DistributedFaceLabeling(labels)
end

function writevtk_local(macro_model::MacroDiscreteModel,filename::String;vtk_kwargs...)
  local_labels  = get_local_face_labeling(macro_model)
  model = macro_model.model
  ranks = linear_indices(local_views(model))

  map(ranks,local_views(model),local_views(local_labels)) do rank, model, labels
    grid   = UnstructuredGrid(get_grid(model))
    _model = UnstructuredDiscreteModel(grid,get_grid_topology(model),labels)
    writevtk(_model,string(filename,"_rank_$(rank)");vtk_kwargs...)
  end
end

function writevtk_global(macro_model::MacroDiscreteModel,filename::String;vtk_kwargs...)
  global_labels  = get_global_face_labeling(macro_model)
  model = macro_model.model

  _models = map(local_views(model),local_views(global_labels)) do model, labels
    grid = UnstructuredGrid(get_grid(model))
    UnstructuredDiscreteModel(grid,get_grid_topology(model),labels)
  end
  _model = DistributedDiscreteModel(_models,get_cell_gids(model))
  
  writevtk(_model,filename;vtk_kwargs...)
end

# Helpers 

# Returns `true` if the face is an interface between two processors, `false` otherwise.
function face_is_interface(cell_owners::Set{<:Integer},rank::Integer)
  # Cases: 
  #  a) Has a single nbor -> Face is at physical boundary or at ghost boundary
  #  b) All nbors are the same -> Face is completely inside or outside my local domain
  #  c) I am not a nbor -> Face is completely outside my local domain
  return (length(cell_owners) != 1) && (rank ∈ cell_owners)
end
face_is_interface(cell_owners::Vector{<:Integer},rank::Integer) = face_is_interface(Set(cell_owners),rank)
