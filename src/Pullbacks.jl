
# These should be the reffes that require a 
# globally-computed pushforward, and thus cannot 
# simply go through the local FESpace constructor.
const PullbackReffes = Union{
  GenericRefFE{RaviartThomas},
  GenericRefFE{Nedelec},
}

function DistributedSingleFieldFESpace(
  model::DistributedDiscreteModel, # Active model, not bg model
  trian::DistributedTriangulation,
  cell_gids::PRange, 
  cell_reffe::AbstractArray{<:AbstractArray{T}};
  labels = get_face_labeling(model), 
  split_own_and_ghost=false, 
  constraint=nothing,
  conformity=nothing,
  scale_dof=false,
  global_meshsize=nothing,
  kwargs...
) where T <: PullbackReffes
  # Construct a globally conforming CellFE
  conf = map(cell_reffe) do cell_reffe
    Conformity(testitem(cell_reffe),conformity)
  end |> getany
  cell_fe = FESpaces.CellFE(model, cell_reffe, conf; scale_dof, global_meshsize)

  spaces = map(
    local_views(model),local_views(trian),local_views(labels), cell_fe
  ) do model, trian, labels, cell_fe
    FESpace(model,cell_fe;trian,labels,kwargs...)
  end

  gids = generate_gids(cell_gids,spaces)
  vector_type = _find_vector_type(spaces,gids;split_own_and_ghost)
  space = DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
  return _add_distributed_constraint(space,cell_reffe,constraint)
end

function FESpaces.CellFE(
  model::DistributedDiscreteModel, cell_reffe::AbstractArray{<:AbstractArray{T}}, conformity::Conformity; kwargs...
) where T <: ReferenceFE
  cell_bases = FESpaces.get_cell_shapefuns_and_dof_basis(
    model, cell_reffe, conformity; kwargs...
  )
  cell_fe = map(local_views(model), cell_reffe, cell_bases) do model, cell_reffe, cell_bases
    cell_shapefuns, cell_dof_basis = cell_bases
    FESpaces.CellFE(model, cell_reffe, cell_shapefuns, cell_dof_basis, conformity)
  end
  return cell_fe
end

function FESpaces.get_cell_shapefuns_and_dof_basis(
  model::DistributedDiscreteModel, cell_reffe::AbstractArray{<:AbstractArray{T}}, conformity::Conformity; kwargs...
) where T <: ReferenceFE
  reffe_name = get_name(T)
  pushforward = Pushforward(reffe_name, conformity)
  cell_Jt = map(local_views(model)) do model
    cell_map = get_cell_map(get_grid(model))
    return lazy_map(Broadcasting(∇), cell_map)
  end
  # The cell changes are the ones that need to be globally consistent
  # They contain sign changes, etc...
  cell_changes = FESpaces.compute_cell_bases_changes(
    reffe_name, pushforward, model, cell_reffe, cell_Jt
  )
  cell_bases = map(
    local_views(model), cell_reffe, cell_changes, cell_Jt
  ) do model, cell_reffe, cell_changes, cell_Jt
    FESpaces.get_cell_shapefuns_and_dof_basis(
      pushforward, model, cell_reffe, cell_changes, cell_Jt; kwargs...
    )
  end
  return cell_bases
end

function FESpaces.compute_cell_bases_changes(
  ::ReferenceFEName, ::ContraVariantPiolaMap, model::DistributedDiscreteModel, cell_reffe, cell_Jt
)
  change = FESpaces.get_sign_flip(model, cell_reffe) # equal to its transposed inverse
  return map(c -> (c, c), change)
end

function FESpaces.compute_cell_bases_changes(
  ::ReferenceFEName, ::CoVariantPiolaMap, model::DistributedDiscreteModel, cell_reffe, cell_Jt
)
  D = num_cell_dims(model)
  poly = only(getany(map(get_polytopes, model)))
  if (D==2) || is_simplex(poly)
    # For these cases, we do not need to aply a sign flip
    return nothing
  elseif (D==3) && is_n_cube(poly)
    change = FESpaces.get_sign_flip(model, cell_reffe)
    return map(c -> (c, c), change)
  end
  @notimplemented
end

function FESpaces.get_sign_flip(model::DistributedDiscreteModel, cell_reffe)
  facet_owners = FESpaces.compute_facet_owners(model)
  map(local_views(model),cell_reffe,facet_owners) do model, cell_reffe, facet_owners
    sign_map = FESpaces.NormalSignMap(model,facet_owners)
    FESpaces.get_sign_flip(model, cell_reffe, sign_map)
  end
end

function FESpaces.compute_facet_owners(model::DistributedDiscreteModel)
  Dc = num_cell_dims(model)
  cell_ids  = partition(get_cell_gids(model))
  facet_ids = partition(get_face_gids(model, Dc-1))
  facet_to_owner = map(FESpaces.compute_facet_owners, local_views(model))

  # Map local owners to global ids
  map(facet_to_owner, cell_ids) do facet_to_owner, cell_ids
    lid_to_gid = local_to_global(cell_ids)
    for f in eachindex(facet_to_owner)
      lid = facet_to_owner[f]
      facet_to_owner[f] = lid_to_gid[lid]
    end
  end

  # Communicate true owners across processes
  wait(consistent!(PVector(facet_to_owner, facet_ids)))

  # Non-local owners will be set to zero, which 
  # will trigger a sign flip (which is the correct behaviour)
  map(facet_to_owner, cell_ids) do facet_to_owner, cell_ids
    gid_to_lid = global_to_local(cell_ids)
    for f in eachindex(facet_to_owner)
      gid = facet_to_owner[f]
      facet_to_owner[f] = gid_to_lid[gid]
    end
  end

  return facet_to_owner
end
