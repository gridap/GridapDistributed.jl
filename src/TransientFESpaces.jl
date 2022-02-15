# Functions for transient FE spaces

function Fields.evaluate!(Ut::DistributedSingleFieldFESpace,U::TransientTrialFESpace,t::Real)
  if isa(U.dirichlet_t,Vector)
    objects_at_t = map( o->o(t), U.dirichlet_t)
  else
    objects_at_t = U.dirichlet_t(t)
  end
  TrialFESpace!(Ut,objects_at_t)
  Ut
end

Fields.evaluate(U::DistributedSingleFieldFESpace,t::Nothing) = U

(U::DistributedSingleFieldFESpace)(t) = U

∂t(U::DistributedSingleFieldFESpace) = HomogeneousTrialFESpace(U)
∂tt(U::DistributedSingleFieldFESpace) = HomogeneousTrialFESpace(U)


# Functions for transient FE Functions

function ODETools.allocate_jacobian(
  op::TransientFETools.TransientFEOperatorFromWeakForm,
  duh::DistributedCellField,
  cache)
  _matdata_jacobians = TransientFETools.fill_initial_jacobians(op,duh)
  matdata = _vcat_distributed_matdata(_matdata_jacobians)
  allocate_matrix(op.assem_t,matdata)
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::TransientFETools.TransientFEOperatorFromWeakForm,
  t::Real,
  xh::TransientDistributedCellField,
  γ::Tuple{Vararg{Real}},
  cache)
  _matdata_jacobians = TransientFETools.fill_jacobians(op,t,xh,γ)
  matdata = _vcat_distributed_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem_t, matdata)
  A
end

function _vcat_distributed_matdata(_matdata)
  term_to_cellmat = map_parts(a->a[1],local_views(_matdata[1]))
  term_to_cellidsrows = map_parts(a->a[2],local_views(_matdata[1]))
  term_to_cellidscols = map_parts(a->a[3],local_views(_matdata[1]))
  for j in 2:length(_matdata)
    term_to_cellmat_j = map_parts(a->a[1],local_views(_matdata[j]))
    term_to_cellidsrows_j = map_parts(a->a[2],local_views(_matdata[j]))
    term_to_cellidscols_j = map_parts(a->a[3],local_views(_matdata[j]))
    term_to_cellmat = map_parts((a,b)->vcat(a,b),local_views(term_to_cellmat),local_views(term_to_cellmat_j))
    term_to_cellidsrows = map_parts((a,b)->vcat(a,b),local_views(term_to_cellidsrows),local_views(term_to_cellidsrows_j))
    term_to_cellidscols = map_parts((a,b)->vcat(a,b),local_views(term_to_cellidscols),local_views(term_to_cellidscols_j))
  end
  map_parts( (a,b,c) -> (a,b,c),
    local_views(term_to_cellmat),
    local_views(term_to_cellidsrows),
    local_views(term_to_cellidscols)
  )
end
