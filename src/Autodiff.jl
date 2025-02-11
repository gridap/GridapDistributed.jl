
const DistributedADTypes = Union{DistributedCellField,DistributedMultiFieldCellField}

# Distributed counterpart of: src/FESpaces/FEAutodiff.jl

function Fields.gradient(f::Function,uh::DistributedADTypes)
  fuh = f(uh)
  FESpaces._gradient(f,uh,fuh)
end

function FESpaces._gradient(f,uh,fuh::DistributedDomainContribution)
  local_terms = map(r -> DomainContribution(), get_parts(fuh))
  local_domains = tuple_of_arrays(map(Tuple∘get_domains,local_views(fuh)))
  for local_trians in local_domains
    g = FESpaces._change_argument(gradient,f,local_trians,uh)
    cell_u = map(FESpaces.get_cell_dof_values,local_views(uh))
    cell_id = map(FESpaces._compute_cell_ids,local_views(uh),local_trians)
    cell_grad = distributed_autodiff_array_gradient(g,cell_u,cell_id)
    map(add_contribution!,local_terms,local_trians,cell_grad)
  end
  DistributedDomainContribution(local_terms)
end

function Fields.jacobian(f::Function,uh::DistributedADTypes)
  fuh = f(uh)
  FESpaces._jacobian(f,uh,fuh)
end

function FESpaces._jacobian(f,uh,fuh::DistributedDomainContribution)
  local_terms = map(r -> DomainContribution(), get_parts(fuh))
  local_domains = tuple_of_arrays(map(Tuple∘get_domains,local_views(fuh)))
  for local_trians in local_domains
    g = FESpaces._change_argument(jacobian,f,local_trians,uh)
    cell_u = map(FESpaces.get_cell_dof_values,local_views(uh))
    cell_id = map(FESpaces._compute_cell_ids,local_views(uh),local_trians)
    cell_grad = distributed_autodiff_array_jacobian(g,cell_u,cell_id)
    map(add_contribution!,local_terms,local_trians,cell_grad)
  end
  DistributedDomainContribution(local_terms)
end

function Fields.hessian(f::Function,uh::DistributedADTypes)
  fuh = f(uh)
  FESpaces._hessian(f,uh,fuh)
end

function FESpaces._hessian(f,uh,fuh::DistributedDomainContribution)
  local_terms = map(r -> DomainContribution(), get_parts(fuh))
  local_domains = tuple_of_arrays(map(Tuple∘get_domains,local_views(fuh)))
  for local_trians in local_domains
    g = FESpaces._change_argument(hessian,f,local_trians,uh)
    cell_u = map(FESpaces.get_cell_dof_values,local_views(uh))
    cell_id = map(FESpaces._compute_cell_ids,local_views(uh),local_trians)
    cell_grad = distributed_autodiff_array_hessian(g,cell_u,cell_id)
    map(add_contribution!,local_terms,local_trians,cell_grad)
  end
  DistributedDomainContribution(local_terms)
end

# There are 4 = 2x2 combinations, coming from:
#   - Creation of the serial CellFields are different for Triangulation and SkeletonTriangulation
#   - Creation of the distributed CellFields are different for DistributedCellField and DistributedMultiFieldCellField
# The internal functions take care of all 4 combinations
function FESpaces._change_argument(op,f,local_trians,uh::DistributedADTypes)
  function dist_cf(uh::DistributedCellField,cfs)
    DistributedCellField(cfs,get_triangulation(uh))
  end
  function dist_cf(uh::DistributedMultiFieldCellField,cfs)
    sf_cfs = map(DistributedCellField,
      [tuple_of_arrays(map(cf -> Tuple(cf.single_fields),cfs))...],
      map(get_triangulation,uh)
    )
    DistributedMultiFieldCellField(sf_cfs,cfs)
  end

  uhs = local_views(uh)
  spaces = map(get_fe_space,uhs)
  function g(cell_u)
    cfs = map(CellField,spaces,cell_u)
    cf = dist_cf(uh,cfs)
    cg = f(cf)
    map(get_contribution,local_views(cg),local_trians)
  end
  g
end

# Distributed counterpart of: src/Arrays/Autodiff.jl
# autodiff_array_xxx

function distributed_autodiff_array_gradient(a,i_to_x)
  dummy_tag = ()->()
  i_to_cfg = map(i_to_x) do i_to_x
    lazy_map(ConfigMap(ForwardDiff.gradient,dummy_tag),i_to_x)
  end
  i_to_xdual = map(i_to_cfg,i_to_x) do i_to_cfg, i_to_x
    lazy_map(DualizeMap(),i_to_cfg,i_to_x)
  end
  i_to_ydual = a(i_to_xdual)
  i_to_result = map(i_to_cfg,i_to_ydual) do i_to_cfg,i_to_ydual
    lazy_map(AutoDiffMap(),i_to_cfg,i_to_ydual)
  end
  return i_to_result
end

function distributed_autodiff_array_jacobian(a,i_to_x)
  dummy_tag = ()->()
  i_to_cfg = map(i_to_x) do i_to_x
    lazy_map(ConfigMap(ForwardDiff.jacobian,dummy_tag),i_to_x)
  end
  i_to_xdual = map(i_to_cfg,i_to_x) do i_to_cfg, i_to_x
    lazy_map(DualizeMap(),i_to_cfg,i_to_x)
  end
  i_to_ydual = a(i_to_xdual)
  i_to_result = map(i_to_cfg,i_to_ydual) do i_to_cfg,i_to_ydual
    lazy_map(AutoDiffMap(),i_to_cfg,i_to_ydual)
  end
  return i_to_result
end

function distributed_autodiff_array_hessian(a,i_to_x)
  agrad = i_to_y -> distributed_autodiff_array_gradient(a,i_to_y)
  distributed_autodiff_array_jacobian(agrad,i_to_x)
end

function distributed_autodiff_array_gradient(a,i_to_x,j_to_i)
  dummy_tag = ()->()
  i_to_cfg = map(i_to_x) do i_to_x
    lazy_map(ConfigMap(ForwardDiff.gradient,dummy_tag),i_to_x)
  end
  i_to_xdual = map(i_to_cfg,i_to_x) do i_to_cfg, i_to_x
    lazy_map(DualizeMap(),i_to_cfg,i_to_x)
  end
  j_to_ydual = a(i_to_xdual)
  j_to_result = map(i_to_cfg,j_to_i,j_to_ydual) do i_to_cfg,j_to_i,j_to_ydual
    j_to_cfg = lazy_map(Reindex(i_to_cfg),j_to_i)
    lazy_map(AutoDiffMap(),j_to_cfg,j_to_ydual)
  end
  return j_to_result
end

function distributed_autodiff_array_jacobian(a,i_to_x,j_to_i)
  dummy_tag = ()->()
  i_to_cfg = map(i_to_x) do i_to_x
    lazy_map(ConfigMap(ForwardDiff.jacobian,dummy_tag),i_to_x)
  end
  i_to_xdual = map(i_to_cfg,i_to_x) do i_to_cfg, i_to_x
    lazy_map(DualizeMap(),i_to_cfg,i_to_x)
  end
  j_to_ydual = a(i_to_xdual)
  j_to_result = map(i_to_cfg,j_to_i,j_to_ydual) do i_to_cfg,j_to_i,j_to_ydual
    j_to_cfg = lazy_map(Reindex(i_to_cfg),j_to_i)
    lazy_map(AutoDiffMap(),j_to_cfg,j_to_ydual)
  end
  return j_to_result
end

function distributed_autodiff_array_hessian(a,i_to_x,i_to_j)
  agrad = i_to_y -> distributed_autodiff_array_gradient(a,i_to_y,i_to_j)
  distributed_autodiff_array_jacobian(agrad,i_to_x,i_to_j)
end

# Skeleton AD

function FESpaces._change_argument(op,f,local_trians::AbstractArray{<:SkeletonTriangulation},uh::DistributedADTypes)
  function dist_cf(uh::DistributedCellField,cfs)
    DistributedCellField(cfs, get_triangulation(uh))
  end
  function dist_cf(uh::DistributedMultiFieldCellField,cfs)
    sf_cfs = map(DistributedCellField,
      [tuple_of_arrays(map(cf -> Tuple(cf.single_fields),cfs))...],
      map(get_triangulation,uh)
    )
    DistributedMultiFieldCellField(sf_cfs,cfs)
  end

  uhs = local_views(uh)
  spaces = map(get_fe_space,uhs)
  function g(cell_u)
    uhs_dual = map(CellField,spaces,cell_u)
    cf_plus  = dist_cf(uh,map(SkeletonCellFieldPair,uhs_dual,uhs))
    cf_minus = dist_cf(uh,map(SkeletonCellFieldPair,uhs,uhs_dual))
    cg_plus  = f(cf_plus)
    cg_minus = f(cf_minus)
    plus  = map(get_contribution,local_views(cg_plus),local_trians)
    minus = map(get_contribution,local_views(cg_minus),local_trians)
    plus, minus
  end
  g
end

function distributed_autodiff_array_gradient(a, i_to_x, j_to_i::AbstractArray{<:SkeletonPair})
  dummy_tag = ()->()
  i_to_cfg = map(i_to_x) do i_to_x
    lazy_map(ConfigMap(ForwardDiff.gradient,dummy_tag),i_to_x)
  end
  i_to_xdual = map(i_to_cfg,i_to_x) do i_to_cfg, i_to_x
    lazy_map(DualizeMap(),i_to_cfg,i_to_x)
  end

  # dual output of both sides at once
  j_to_ydual_plus, j_to_ydual_minus = a(i_to_xdual)

  j_to_result = map(i_to_cfg,j_to_i,j_to_ydual_plus,j_to_ydual_minus) do i_to_cfg,j_to_i,j_to_ydual_plus,j_to_ydual_minus
    # Work for plus side
    j_to_cfg_plus = lazy_map(Reindex(i_to_cfg),j_to_i.plus)
    j_to_result_plus = lazy_map(AutoDiffMap(),j_to_cfg_plus,j_to_ydual_plus)

    # Work for minus side
    j_to_cfg_minus = lazy_map(Reindex(i_to_cfg),j_to_i.minus)
    j_to_result_minus = lazy_map(AutoDiffMap(),j_to_cfg_minus,j_to_ydual_minus)

    # Assemble on SkeletonTriangulation expects an array of interior of facets
    # where each entry is a 2-block BlockVector with the first block being the
    # contribution of the plus side and the second, the one of the minus side
    is_single_field = eltype(eltype(j_to_result_plus)) <: Number
    k = is_single_field ? BlockMap(2,[1,2]) : Fields.BlockBroadcasting(BlockMap(2,[1,2]))
    lazy_map(k,j_to_result_plus,j_to_result_minus)
  end

  return j_to_result
end

function distributed_autodiff_array_jacobian(a, i_to_x, j_to_i::AbstractArray{<:SkeletonPair})
  dummy_tag = ()->()
  i_to_cfg = map(i_to_x) do i_to_x
    lazy_map(ConfigMap(ForwardDiff.jacobian,dummy_tag),i_to_x)
  end
  i_to_xdual = map(i_to_cfg,i_to_x) do i_to_cfg, i_to_x
    lazy_map(DualizeMap(),i_to_cfg,i_to_x)
  end

  # dual output of both sides at once
  j_to_ydual_plus, j_to_ydual_minus = a(i_to_xdual)

  j_to_result = map(i_to_cfg,j_to_i,j_to_ydual_plus,j_to_ydual_minus) do i_to_cfg,j_to_i,j_to_ydual_plus,j_to_ydual_minus
    # Work for plus side
    j_to_cfg_plus = lazy_map(Reindex(i_to_cfg),j_to_i.plus)
    j_to_result_plus = lazy_map(AutoDiffMap(),j_to_cfg_plus,j_to_ydual_plus)

    # Work for minus side
    j_to_cfg_minus = lazy_map(Reindex(i_to_cfg),j_to_i.minus)
    j_to_result_minus = lazy_map(AutoDiffMap(),j_to_cfg_minus,j_to_ydual_minus)

    # Merge the columns into a 2x2 block matrix
    # I = [[(CartesianIndex(i,),CartesianIndex(i,j)) for i in 1:2] for j in 1:2]
    I = [
      [(CartesianIndex(1,), CartesianIndex(1, 1)), (CartesianIndex(2,), CartesianIndex(2, 1))], # Plus  -> First column
      [(CartesianIndex(1,), CartesianIndex(1, 2)), (CartesianIndex(2,), CartesianIndex(2, 2))]  # Minus -> Second column
    ]
    is_single_field = eltype(eltype(j_to_result_plus)) <: AbstractArray
    k = is_single_field ? Fields.MergeBlockMap((2,2),I) : Fields.BlockBroadcasting(Fields.MergeBlockMap((2,2),I))
    lazy_map(k,j_to_result_plus,j_to_result_minus)
  end

  return j_to_result
end
