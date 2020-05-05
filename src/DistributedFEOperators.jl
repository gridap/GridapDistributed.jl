
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

