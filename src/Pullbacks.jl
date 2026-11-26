
using Gridap.Polynomials: RotationCache

# The push-forwards for which this file provides a distributed method.
const PiolaMaps = Union{IdentityPiolaMap,ContraVariantPiolaMap,CoVariantPiolaMap}

"""
    needs_distributed_cell_bases_changes(name::ReferenceFEName, push::Pushforward)
    needs_distributed_cell_bases_changes(cell_reffe::AbstractArray, conf::Conformity)

Whether `compute_cell_bases_changes` for this reference FE needs data no rank
can get from its local view -- globally agreed facet owners, or global vertex
ids. Either way the space is one `FESpace` per rank; what this decides is
whether their common `FESpaces.CellFE` is built for the whole distributed model
first, or each rank builds its own.
"""
needs_distributed_cell_bases_changes(::ReferenceFEName, ::Pushforward) = false

# Contravariant facet moments use the outward normal that needs an owner every rank agrees on
needs_distributed_cell_bases_changes(::ReferenceFEName, ::ContraVariantPiolaMap) = true
# Built from face permutations rather than the outward normal
needs_distributed_cell_bases_changes(::MardalTaiWinther, ::ContraVariantPiolaMap) = false

# These need globally agreed physical vertices IDs.
needs_distributed_cell_bases_changes(
  ::Union{RotatingPΛName,TrimmedPΛName}, ::CoVariantPiolaMap
) = true

function needs_distributed_cell_bases_changes(
  ::CartProd{K,NS}, pf::ReferenceFEs.CartProdPushforward{K}
) where {K,NS}
  names = ReferenceFEs._cp_factor_names(Val(K), NS)
  pushs = ReferenceFEs._cp_maps(pf, Val(K))
  return any(c -> needs_distributed_cell_bases_changes(names[c](), pushs[c]), 1:K)
end

needs_distributed_cell_bases_changes(cell_reffe::AbstractArray, conf::Conformity) =
  needs_distributed_cell_bases_changes(eltype(getany(cell_reffe)), conf)
needs_distributed_cell_bases_changes(::Type{<:GenericRefFE{Name}}, conf::Conformity) where Name =
  needs_distributed_cell_bases_changes(Name(), Pushforward(Name(), conf))
needs_distributed_cell_bases_changes(::Type{<:ReferenceFE}, ::Conformity) = false

function DistributedSingleFieldFESpace(
  model::DistributedDiscreteModel, # Active model, not bg model
  trian::DistributedTriangulation,
  cell_gids::PRange,
  cell_reffe::AbstractArray;
  conformity=nothing,
  split_own_and_ghost=false,
  constraint=nothing,
  kwargs...
)
  conf = map(cell_reffe) do cell_reffe
    Conformity(testitem(cell_reffe),conformity)
  end |> getany

  # One local FESpace per rank, contructed using rank local info or not.
  spaces = if needs_distributed_cell_bases_changes(cell_reffe,conf)
    _nonlocal_spaces(model,trian,cell_reffe,conf;kwargs...)
  else
    map(local_views(model),local_views(trian),cell_reffe) do model, trian, cell_reffe
      FESpace(model,cell_reffe;trian,conformity=conf,kwargs...)
    end
  end

  gids = generate_gids(cell_gids,spaces)
  vector_type = _find_vector_type(spaces,gids;split_own_and_ghost)
  space = DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
  return _add_distributed_constraint(space,cell_reffe,constraint)
end

function _nonlocal_spaces( model, trian, cell_reffe, conf;
  labels = get_face_labeling(model), scale_dof=false, global_meshsize=nothing, kwargs...
)
  # Construct a globally conforming CellFE
  cell_fe = FESpaces.CellFE(model, cell_reffe, conf; scale_dof, global_meshsize)

  map(
    local_views(model),local_views(trian),local_views(labels), cell_fe
  ) do model, trian, labels, cell_fe
    FESpace(model,cell_fe;trian,labels,kwargs...)
  end
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

# compute_cell_bases_changes supplying the quantity(ies) not determined by
# local infos, and delegates the rest to Gridap.

function FESpaces.compute_cell_bases_changes(
  ::ReferenceFEName, ::IdentityPiolaMap, model::DistributedDiscreteModel, cell_reffe, cell_Jt
)
  map(_ -> nothing, cell_reffe)
end

function FESpaces.compute_cell_bases_changes(
  ::ReferenceFEName, ::ContraVariantPiolaMap, model::DistributedDiscreteModel, cell_reffe, cell_Jt
)
  # The outward-normal flip is the global part (global owning facet needed)
  normal_change = FESpaces.get_sign_flip(model, cell_reffe) # equal to its transposed inverse
  map(local_views(model), cell_reffe, normal_change) do model, cell_reffe, normal_change
    FESpaces._face_sign_cell_bases_changes(model, cell_reffe, normal_change)
  end
end

function FESpaces.compute_cell_bases_changes(
  name::ReferenceFEName, push::CoVariantPiolaMap, model::DistributedDiscreteModel, cell_reffe, cell_Jt
)
  # nothing global
  map(local_views(model), cell_reffe, cell_Jt) do model, cell_reffe, cell_Jt
    FESpaces.compute_cell_bases_changes(name, push, model, cell_reffe, cell_Jt)
  end
end

# `rotating_pλ` / `trimmed_pλ` this change depends on cell's vertex ids global order.
# Those ids are global in serial but local in a distributed view, so each rank
# feeds the serial kernel its global vertex ids instead.
function FESpaces.compute_cell_bases_changes(
  ::Union{RotatingPΛName,TrimmedPΛName}, ::CoVariantPiolaMap,
  model::DistributedDiscreteModel, cell_reffe, cell_Jt
)
  D = num_cell_dims(model)
  vertex_gids = partition(get_face_gids(model, 0))
  map(local_views(model), cell_reffe, vertex_gids) do model, cell_reffe, vertex_gids
    lcell_verts = Table(get_faces(get_grid_topology(model), D, 0))
    l2g = local_to_global(vertex_gids)
    gcell_verts = Table(l2g[lcell_verts.data], lcell_verts.ptrs)
    rc = RotationCache(get_prebasis(testitem(cell_reffe)))
    FESpaces.compute_pλ_cell_bases_changes(gcell_verts, rc)
  end
end

function FESpaces.compute_cell_bases_changes(
  ::CartProd{K,NS}, push::ReferenceFEs.CartProdPushforward{K},
  model::DistributedDiscreteModel, cell_reffe, cell_Jt
) where {K,NS}

  pushs = ReferenceFEs._cp_maps(push, Val(K))
  @notimplementedif !all(pf -> isa(pf, PiolaMaps), pushs) """\n
  A Cartesian product reaching the distributed FESpace constructor may only have
  Piola-mapped factors, got $(pushs). The other factors have permutation-based
  changes of basis, for which Gridap provides no distributed method.
  """
  reffes = map(cell_reffe -> ReferenceFEs.get_metadata(testitem(cell_reffe)), cell_reffe) |> getany

  # Recurse with the distributed model, for factors needing global data to get it
  changes = ntuple(K) do c
    factor_reffe = map(cell_reffe -> Fill(reffes[c], length(cell_reffe)), cell_reffe)
    FESpaces.compute_cell_bases_changes(
      get_name(reffes[c]), pushs[c], model, factor_reffe, cell_Jt
    )
  end

  map(cell_reffe, changes...) do cell_reffe, changes...
    all(isnothing, changes) && return nothing
    eye(c) = Fill(Matrix{Float64}(I, num_dofs(reffes[c]), num_dofs(reffes[c])), length(cell_reffe))
    fwd = ntuple(c -> isnothing(changes[c]) ? eye(c) : changes[c][1], K)
    bwd = ntuple(c -> isnothing(changes[c]) ? eye(c) : changes[c][2], K)
    return (lazy_map(ReferenceFEs.CartProdBlockDiag(), fwd...),
            lazy_map(ReferenceFEs.CartProdBlockDiag(), bwd...))
  end
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

  function select_nbor_with_max_gid(nbor_lids, lid_to_gid)
    max_lid, max_gid = -1, -1
    for lid in nbor_lids
      gid = lid_to_gid[lid]
      if gid > max_gid
        max_lid, max_gid = lid, gid
      end
    end
    return max_lid
  end
  facet_to_owner = map(local_views(model), cell_ids) do model, cell_ids
    lid_to_gid = local_to_global(cell_ids)
    select_nbor = Base.Fix2(select_nbor_with_max_gid, lid_to_gid)
    return FESpaces.compute_facet_owners(model, select_nbor)
  end

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
