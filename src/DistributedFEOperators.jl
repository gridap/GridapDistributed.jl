
"""
"""
function Gridap.FESpaces.AffineFEOperator(trial::DistributedFESpace,
                                          test::DistributedFESpace,
                                          matrix::AbstractMatrix,
                                          vector::AbstractVector)
  @assert false # TO-DO
end

"""
"""
function Gridap.FESpaces.AffineFEOperator(weakform::Function,
                                          trial::DistributedFESpace,
                                          test::DistributedFESpace,
                                          assem::DistributedAssembler)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  uhd = zero(trial)
  matcontribs, veccontribs = weakform(u,v)
  data = collect_cell_matrix_and_vector(trial,test,matcontribs,veccontribs,uhd)
  A,b = assemble_matrix_and_vector(assem,data)
  op = AffineOperator(A,b)
  AffineFEOperator(trial,test,op)
end


function Gridap.FESpaces.FEOperator(assem::DistributedAssembler,terms::DistributedData)
  trial = assem.trial
  test = assem.test
  FEOperatorFromTerms(trial,test,assem,terms)
end
