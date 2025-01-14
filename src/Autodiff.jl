
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
    cell_u = map(FESpaces._get_cell_dof_values,local_views(uh),local_trians)
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
    cell_u = map(FESpaces._get_cell_dof_values,local_views(uh),local_trians)
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
    cell_u = map(FESpaces._get_cell_dof_values,local_views(uh),local_trians)
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
  serial_cf(::Triangulation,space,cell_u) = CellField(space,cell_u)
  serial_cf(::SkeletonTriangulation,space,cell_u) = SkeletonCellFieldPair(space,cell_u) 
  function dist_cf(uh::DistributedCellField,trians,spaces,cell_u)
    DistributedCellField(
      map(serial_cf,trians,spaces,cell_u),
      get_triangulation(uh)
    )
  end
  function dist_cf(uh::DistributedMultiFieldCellField,trians,spaces,cell_u)
    mf_cfs = map(serial_cf,trians,spaces,cell_u)
    sf_cfs = map(DistributedCellField,
      [tuple_of_arrays(map(cf -> Tuple(cf.single_fields),mf_cfs))...],
      map(get_triangulation,uh)
    )
    DistributedMultiFieldCellField(sf_cfs,mf_cfs)
  end

  spaces = map(get_fe_space,local_views(uh))
  function g(cell_u)
    cf = dist_cf(uh,local_trians,spaces,cell_u)
    cell_grad = f(cf)
    map(get_contribution,local_views(cell_grad),local_trians)
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
