
function Fields.gradient(f::Function,uh::DistributedCellField)
  fuh = f(uh)
  FESpaces._gradient(f,uh,fuh)
end

function FESpaces._gradient(f,uh,fuh::DistributedDomainContribution)
  local_terms = map(r -> DomainContribution(), get_parts(fuh))
  local_domains = tuple_of_arrays(map(Tuple∘get_domains,local_views(fuh)))
  for local_trians in local_domains
    g = FESpaces._change_argument(gradient,f,local_trians,uh)
    cell_u = map(get_cell_dof_values,local_views(uh))
    cell_id = map(FESpaces._compute_cell_ids,local_views(uh),local_trians)
    cell_grad = distributed_autodiff_array_gradient(g,cell_u,cell_id)
    map(add_contribution!,local_terms,local_trians,cell_grad)
  end
  DistributedDomainContribution(local_terms)
end

function Fields.jacobian(f::Function,uh::DistributedCellField)
  fuh = f(uh)
  FESpaces._jacobian(f,uh,fuh)
end

function FESpaces._jacobian(f,uh,fuh::DistributedDomainContribution)
  local_terms = map(r -> DomainContribution(), get_parts(fuh))
  local_domains = tuple_of_arrays(map(Tuple∘get_domains,local_views(fuh)))
  for local_trians in local_domains
    g = FESpaces._change_argument(jacobian,f,local_trians,uh)
    cell_u = map(get_cell_dof_values,local_views(uh))
    cell_id = map(FESpaces._compute_cell_ids,local_views(uh),local_trians)
    cell_grad = distributed_autodiff_array_jacobian(g,cell_u,cell_id)
    map(add_contribution!,local_terms,local_trians,cell_grad)
  end
  DistributedDomainContribution(local_terms)
end

function FESpaces._change_argument(op,f,trian,uh::DistributedCellField)
  spaces = map(get_fe_space,local_views(uh))
  function g(cell_u)
    cf = DistributedCellField(
      map(CellField,spaces,cell_u),
      get_triangulation(uh),
    )
    cell_grad = f(cf)
    map(get_contribution,local_views(cell_grad),trian)
  end
  g
end

# autodiff_array_xxx

function distributed_autodiff_array_gradient(a,i_to_x)
  dummy_forwarddiff_tag = ()->()
  i_to_xdual = map(i_to_x) do i_to_x
    lazy_map(DualizeMap(ForwardDiff.gradient,dummy_forwarddiff_tag),i_to_x)
  end
  i_to_ydual = a(i_to_xdual)
  i_to_result = map(i_to_ydual,i_to_x) do i_to_ydual,i_to_x
    i_to_cfg = lazy_map(ConfigMap(ForwardDiff.gradient,dummy_forwarddiff_tag),i_to_x)
    lazy_map(AutoDiffMap(ForwardDiff.gradient),i_to_ydual,i_to_x,i_to_cfg)
  end
  return i_to_result
end

function distributed_autodiff_array_jacobian(a,i_to_x)
  dummy_forwarddiff_tag = ()->()
  i_to_xdual = map(i_to_x) do i_to_x
    lazy_map(DualizeMap(ForwardDiff.jacobian,dummy_forwarddiff_tag),i_to_x)
  end
  i_to_ydual = a(i_to_xdual)
  i_to_result = map(i_to_ydual,i_to_x) do i_to_ydual,i_to_x
    i_to_cfg = lazy_map(ConfigMap(ForwardDiff.jacobian,dummy_forwarddiff_tag),i_to_x)
    lazy_map(AutoDiffMap(ForwardDiff.jacobian),i_to_ydual,i_to_x,i_to_cfg)
  end
  i_to_result
end

function distributed_autodiff_array_gradient(a,i_to_x,j_to_i)
  dummy_forwarddiff_tag = ()->()
  i_to_xdual = map(i_to_x) do i_to_x
    lazy_map(DualizeMap(ForwardDiff.gradient,dummy_forwarddiff_tag),i_to_x)
  end
  j_to_ydual = a(i_to_xdual)
  j_to_result = map(i_to_x,j_to_i,j_to_ydual) do i_to_x,j_to_i,j_to_ydual
    j_to_x = lazy_map(Reindex(i_to_x),j_to_i)
    j_to_cfg = lazy_map(ConfigMap(ForwardDiff.gradient,dummy_forwarddiff_tag),j_to_x)
    lazy_map(AutoDiffMap(ForwardDiff.gradient),j_to_ydual,j_to_x,j_to_cfg)
  end
  return j_to_result
end

function distributed_autodiff_array_jacobian(a,i_to_x,j_to_i)
  dummy_forwarddiff_tag = ()->()
  i_to_xdual = map(i_to_x) do i_to_x
    lazy_map(DualizeMap(ForwardDiff.jacobian,dummy_forwarddiff_tag),i_to_x)
  end
  j_to_ydual = a(i_to_xdual)
  j_to_result = map(i_to_x,j_to_i,j_to_ydual) do i_to_x,j_to_i,j_to_ydual
    j_to_x = lazy_map(Reindex(i_to_x),j_to_i)
    j_to_cfg = lazy_map(ConfigMap(ForwardDiff.jacobian,dummy_forwarddiff_tag),j_to_x)
    lazy_map(AutoDiffMap(ForwardDiff.jacobian),j_to_ydual,j_to_x,j_to_cfg)
  end
  j_to_result
end
