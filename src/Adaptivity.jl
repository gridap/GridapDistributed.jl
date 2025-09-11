
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
  _new_model = CartesianDiscreteModel(pdesc)

  map_main(new_ranks) do r
    @debug "Redistributing DistributedCartesianModel:
      > Old: $(repr("text/plain",old_model.metadata))
      > New: $(repr("text/plain",_new_model.metadata))
    "
    msg1 = "Both models should have the same number of cells for redistribution!"
    @check old_model.metadata.descriptor.partition == desc.partition msg1
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
      old_cell_to_face_entity,new_cell_to_face_ids,new_model,glue;T=Int32
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

# Coarsening for polytopal meshes
#
# We assume some properties of the coarsening:
#   - The patch topology must the a partition of the owned fine cells, i.e 
#       + All patches are disjoint
#       + All owned fine cells are part of a patch
#   - All patches are fully owned by a single processor. I.e we do NOT allow 
#     patches that are split between multiple processors.
# The second property is slightly restrictive, but I think it is very reasonable... allowing 
# for split patches would add an insane amount of complexity to the code. Basically, 
# this property means that we can always coarsen the owned fine cells locally, then 
# communicate to find out ghosts. 
# If we ever need to have split patches, one should instead re-distribute the model accordingly
# and then coarsen.
#
# The main idea of the algorithm is as follows:
#   1. First, we coarsen the owned part of the models
#   2. We then create a global numering of the coarse cells
#   3. We can use the above to build the cell-to-node connectivity
#   4. We also need a global numbering of the coarse nodes, which we need to 
#      communicate the model coordinates
#   5. We can then create our coarse model
function Adaptivity.coarsen(fmodel::DistributedDiscreteModel, ptopo::DistributedPatchTopology; return_glue=false)

  # First, some preliminary checks
  fgids = partition(get_cell_gids(fmodel))
  map(local_views(ptopo), fgids) do ptopo, fids
    patch_cells = Geometry.get_patch_cells(ptopo)
    lcell_to_ocell = local_to_own(fids)
    ocell_to_lcell = own_to_local(fids)
    @check allunique(patch_cells.data) "Patches must not overlap"
    @check all(c -> !iszero(lcell_to_ocell[c]), patch_cells.data) "All local patches must be fully owned"
    @check issubset(ocell_to_lcell, patch_cells.data) "All owned cells must be part of a patch"
  end

  # 1. Local coarsening on the owned part of the model
  own_polys, own_connectivity = map(local_views(fmodel), local_views(ptopo)) do fmodel, ptopo
    Adaptivity.generate_patch_polytopes(fmodel,ptopo)
  end |> tuple_of_arrays

  # 2. Coarse cell gids
  #  - Each processor can easily create the global numbering of their owned coarse cells.
  #  - Through the fine cell ghosts, we have to communicate the ghost coarse cell gids, and
  #    from that get the local-to-global mapping.
  Dc = num_cell_dims(fmodel)
  fcell_gids = partition(get_face_gids(fmodel, Dc))
  n_own_ccells = map(Geometry.num_patches, local_views(ptopo))
  n_cgids = sum(n_own_ccells)
  first_cgid = scan(+,n_own_ccells,type=:exclusive,init=1)
  fcell_to_cgid = map(local_views(ptopo), fcell_gids, first_cgid) do ptopo, fcell_gids, first_cgid
    patch_cells = Geometry.get_patch_cells(ptopo)
    fcell_to_cgid = zeros(Int, local_length(fcell_gids))
    Arrays.flatten_partition!(fcell_to_cgid,patch_cells)
    fcell_to_cgid .+= first_cgid - 1
    return fcell_to_cgid
  end
  consistent!(PVector(fcell_to_cgid, fcell_gids)) |> wait

  ccell_gids, fcell_to_ccell = map(fcell_gids, fcell_to_cgid, first_cgid, n_own_ccells) do fids, fcell_to_cgid, first_cgid, n_own_ccells
    owner = part_id(fids)
    own_range = first_cgid:(first_cgid+n_own_ccells-1)
    c_own_to_global = collect(Int, own_range)
    c_own_to_flid  = collect(Int32, indexin(c_own_to_global, fcell_to_cgid))

    is_ghost(gid) = !iszero(gid) && (gid ∉ own_range)
    c_ghost_to_global = filter!(is_ghost, unique(fcell_to_cgid))
    c_ghost_to_flid  = collect(Int32, indexin(c_ghost_to_global, fcell_to_cgid))
    c_ghost_to_owner = local_to_owner(fids)[c_ghost_to_flid]

    own_cgids = OwnIndices(n_cgids, owner, c_own_to_global)
    ghost_cgids = GhostIndices(n_cgids, c_ghost_to_global, c_ghost_to_owner)
    cgids = OwnAndGhostIndices(own_cgids, ghost_cgids)

    cgid_to_clid = global_to_local(cgids)
    fcell_to_ccell = zeros(Int32, local_length(fids))
    for (flid, cgid) in enumerate(fcell_to_cgid)
      fcell_to_ccell[flid] = cgid_to_clid[cgid]
    end
    return cgids, fcell_to_ccell
  end |> tuple_of_arrays

  # 3 & 4. Coarse node gids:
  #  - Each coarse node corresponds to a fine node. However: not all local coarse nodes 
  #    are also local fine nodes. Therefore we will have to work with the 
  #    global fine node ids to have a unique numbering.
  #  - We will communicate the coarse-cell-to-fine-node-gid connectivity, then 
  #    build a coarse node numbering from that.

  # First: We communicate the number of nodes per coarse cell
  ccell_to_nnodes = map(ccell_gids, own_connectivity) do ccids, own_connectivity
    nnodes = zeros(Int32, local_length(ccids))
    nnodes[own_to_local(ccids)] .= map(length, own_connectivity)
    return nnodes
  end
  consistent!(PVector(ccell_to_nnodes, ccell_gids)) |> wait

  # We communicate the connectivity: 
  # For each coarse cell, the fine node gids in that cell
  fnode_gids = partition(get_face_gids(fmodel, 0))
  connectivity = map(
    own_connectivity, ccell_to_nnodes, ccell_gids, fnode_gids
  ) do own_connectivity, ccell_to_nnodes, ccids, fnids
    fnode_local_to_global = local_to_global(fnids)
    ptrs = pushfirst!(ccell_to_nnodes, 0)
    Arrays.length_to_ptrs!(ptrs)  
    data = zeros(Int32, ptrs[end]-1)
    for (oid, lid) in enumerate(own_to_local(ccids))
      node_lids = view(own_connectivity, oid)
      node_gids = view(fnode_local_to_global,node_lids)
      data[ptrs[lid]:(ptrs[lid+1]-1)] .= node_gids
    end
    return JaggedArray(data, ptrs)
  end
  consistent!(PVector(connectivity, ccell_gids)) |> wait
  connectivity = map(c -> Table(c.data, c.ptrs), connectivity)

  # We create a local numbering of the coarse nodes and renumber the connectivity.
  # For later, we also return the local-to-local mapping of the fine nodes to coarse nodes.
  n_cnodes, fnode_to_cnode = map(fnode_gids, connectivity) do fnids, conn
    n_lid = 0
    fgid_to_clid = Dict{Int,Int32}()
    for k in eachindex(conn.data)
      fgid = conn.data[k]
      clid = get!(fgid_to_clid, fgid, n_lid + 1)
      n_lid += (clid == n_lid + 1) # Increment if it's new
      conn.data[k] = clid # Renumber the connectivity
    end
    fnode_to_cnode = zeros(Int32, local_length(fnids))
    for (flid, fgid) in enumerate(local_to_global(fnids))
      fnode_to_cnode[flid] = get!(fgid_to_clid,fgid,0)
    end
    return n_lid, fnode_to_cnode
  end |> tuple_of_arrays

  # Finally, we use pre-existing routines to generate the coarse node gids
  cnode_gids = generate_gids(PRange(ccell_gids), connectivity, n_cnodes) |> partition

  # Coarse node coordinates:
  cnode_coords = map(local_views(fmodel), cnode_gids, fnode_to_cnode) do fmodel, cngids, fnode_to_cnode
    @check issubset(own_to_local(cngids), fnode_to_cnode)
    fnode_coords = Geometry.get_vertex_coordinates(get_grid_topology(fmodel))
    cnode_coords = Vector{eltype(fnode_coords)}(undef, local_length(cngids))
    for (flid, clid) in enumerate(fnode_to_cnode)
      if clid > 0
        cnode_coords[clid] = fnode_coords[flid] # Fill owned info
      end
    end
    return cnode_coords
  end
  consistent!(PVector(cnode_coords, cnode_gids)) |> wait # Communicate coarse coords

  # Coarse polytopes: 
  # - We already have the owned polytopes from the local coarsening
  # - We create the ghost polytopes from the connectivity and the coarse node coordinates
  # TODO: The creation of 3D polyhedra requires a bit more information! This is the only thing missing for 3D.
  @notimplementedif Dc != 2 "Coarsening is only implemented for 2D polytopal meshes"
  polys = map(ccell_gids, connectivity, own_polys, cnode_coords) do ccids, conn, own_polys, cnode_coords
    polys = Vector{eltype(own_polys)}(undef, local_length(ccids))
    for (lid, oid) in enumerate(local_to_own(ccids))
      if oid > 0
        polys[lid] = own_polys[oid] # Owned polytope
      else
        polys[lid] = Polygon(cnode_coords[view(conn, lid)]) # Ghost polytope
      end
    end
    return polys
  end

  # 5. We can finally create our coarse model
  face_gids = Vector{PRange}(undef, Dc+1)
  face_gids[end] = PRange(ccell_gids)
  face_gids[1] = PRange(cnode_gids)
  ctopo = GridapDistributed.DistributedGridTopology(
    map(Geometry.PolytopalGridTopology, cnode_coords, connectivity, polys), face_gids
  )
  _setup_consistent_faces!(ctopo)
  labels = Geometry.FaceLabeling(ctopo)
  cmodels = map(local_views(ctopo), local_views(labels)) do ctopo, labels
    cgrid = Geometry.PolytopalGrid(ctopo)
    Geometry.PolytopalDiscreteModel(cgrid, ctopo, labels)
  end
  cmodel = GridapDistributed.DistributedDiscreteModel(cmodels, face_gids)
  (!return_glue) && (return cmodel)

  # If required, create the adaptivity glues
  ftopo = get_grid_topology(fmodel)
  glues = map(
    ccell_gids, cnode_gids, fcell_to_ccell, fnode_to_cnode, local_views(ftopo), local_views(ctopo)
  ) do ccids, cnids, fcell_to_ccell, fnode_to_cnode, ftopo, ctopo
    ccell_to_fcell = Arrays.inverse_table(fcell_to_ccell, local_length(ccids))
    cnode_to_fnode = find_inverse_index_map(fnode_to_cnode, local_length(cnids))
    Adaptivity.generate_patch_adaptivity_glue(
      ftopo, ctopo, fcell_to_ccell, ccell_to_fcell, fnode_to_cnode, cnode_to_fnode,
    )
  end
  return cmodel, glues
end
