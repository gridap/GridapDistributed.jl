"""
    add_entry!(A,v,i,j,combine=+)

Add an entry given its position and the operation to perform.
This method implementation (at present) assumes the following:
  (1) i,j are local identifiers, and A has been set up a LocalToGlobalMapping IS.
  (2) The insertion mode of A is PETSC.ADD_VALUES
"""
function Gridap.Algebra.add_entry!(A::PETSc.Mat{Float64},
                                   v,
                                   i::Integer,
                                   j::Integer)
  PETSc.set_values_local!(A,
                          PETSc.PetscInt[i-1],
                          PETSc.PetscInt[j-1],
                          Float64[v])
end

"""
    add_entry!(A,v,i,combine=+)

Add an entry given its position and the operation to perform.
This method implementation (at present) assumes the following:
  (1) i,j are local identifiers, and A has been set up a LocalToGlobalMapping IS.
  (2) The insertion mode of A is PETSC.ADD_VALUES
"""
function Gridap.Algebra.add_entry!(A::PETSc.Vec{Float64},
                                   v,
                                   i::Integer)
  PETSc.set_values_local!(A,
                          PETSc.PetscInt[i-1],
                          Float64[v])
end