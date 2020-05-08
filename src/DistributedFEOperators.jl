
function Gridap.FESpaces.AffineFEOperator(dassem::DistributedAssembler, dterms)

  trial = dassem.trial
  test = dassem.test

  u = get_cell_basis(trial)
  v = get_cell_basis(test)

  uhd = zero(trial)

  data = collect_cell_matrix_and_vector(uhd,u,v,dterms)
  A,b = assemble_matrix_and_vector(dassem,data)

  op = AffineOperator(A,b)
  AffineFEOperator(trial,test,op)

end

function Gridap.FESpaces.FEOperator(assem::DistributedAssembler,terms::DistributedData)
  trial = assem.trial
  test = assem.test
  FEOperatorFromTerms(trial,test,assem,terms)
end

