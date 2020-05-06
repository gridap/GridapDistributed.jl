
function Gridap.FESpaces.AffineFEOperator(dassem::DistributedAssembler, dterms)

  dvecdata = DistributedData(dassem,dterms) do part, assem, terms
    U = get_trial(assem)
    V = get_test(assem)
    u0 = zero(U)
    v = get_cell_basis(V)
    collect_cell_vector(u0,v,terms)
  end

  dmatdata = DistributedData(dassem,dterms) do part, assem, terms
    U = get_trial(assem)
    V = get_test(assem)
    u = get_cell_basis(U)
    v = get_cell_basis(V)
    collect_cell_matrix(u,v,terms)
  end

  A = assemble_matrix(dassem,dmatdata)
  b = assemble_vector(dassem,dvecdata)
  trial = dassem.trial
  test = dassem.test

  op = AffineOperator(A,b)
  AffineFEOperator(trial,test,op)

end

struct DistributedFEOperatorFromTerms <: FEOperator
  assem::DistributedAssembler
  terms::DistributedData
end

function Gridap.FESpaces.FEOperator(assem::DistributedAssembler,terms::DistributedData)
  DistributedFEOperatorFromTerms(assem,terms)
end

function Gridap.FESpaces.get_test(op::DistributedFEOperatorFromTerms)
  op.assem.test
end

function Gridap.FESpaces.get_trial(op::DistributedFEOperatorFromTerms)
  op.assem.trial
end

function Gridap.Algebra.allocate_residual(op::DistributedFEOperatorFromTerms,uh)
  @assert is_a_fe_function(uh)
  dvecdata = DistributedData(op.assem,uh,op.terms) do part, assem, uh, terms
    V = get_test(assem)
    v = get_cell_basis(V)
    collect_cell_residual(uh,v,terms)
  end
  allocate_vector(op.assem,dvecdata)
end

function Gridap.Algebra.residual!(b::AbstractVector,op::DistributedFEOperatorFromTerms,uh)
  @assert is_a_fe_function(uh)
  dvecdata = DistributedData(op.assem,uh,op.terms) do part, assem, uh, terms
    V = get_test(assem)
    v = get_cell_basis(V)
    collect_cell_residual(uh,v,terms)
  end
  assemble_vector!(b,op.assem,dvecdata)
end

function Gridap.Algebra.allocate_jacobian(op::DistributedFEOperatorFromTerms,uh)
  @assert is_a_fe_function(uh)
  dmatdata = DistributedData(op.assem,uh,op.terms) do part, assem, uh, terms
    U = get_trial(assem)
    du = get_cell_basis(U)
    V = get_test(assem)
    v = get_cell_basis(V)
    collect_cell_jacobian(uh,du,v,terms)
  end
  allocate_matrix(op.assem,dmatdata)
end

function Gridap.Algebra.jacobian!(A::AbstractMatrix,op::DistributedFEOperatorFromTerms,uh)
  @assert is_a_fe_function(uh)
  dmatdata = DistributedData(op.assem,uh,op.terms) do part, assem, uh, terms
    U = get_trial(assem)
    du = get_cell_basis(U)
    V = get_test(assem)
    v = get_cell_basis(V)
    collect_cell_jacobian(uh,du,v,terms)
  end
  assemble_matrix!(A,op.assem,dmatdata)
end


