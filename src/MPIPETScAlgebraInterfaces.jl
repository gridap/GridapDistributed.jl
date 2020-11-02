"""
    add_entry!(A,v,i,j,combine=+)

Add an entry given its position and the operation to perform.
This method implementation (at present) assumes the following:
  (1) i,j are local identifiers, and A has been set up a LocalToGlobalMapping IS.
  (2) The insertion mode of A is GridapDistributedPETScWrappers.ADD_VALUES
"""
function Gridap.Algebra.add_entry!(A::GridapDistributedPETScWrappers.Mat{Float64},
                                   v,
                                   i::Integer,
                                   j::Integer)
  GridapDistributedPETScWrappers.set_values_local!(A,
                          GridapDistributedPETScWrappers.PetscInt[i-1],
                          GridapDistributedPETScWrappers.PetscInt[j-1],
                          Float64[v])
end

"""
    add_entry!(A,v,i,combine=+)

Add an entry given its position and the operation to perform.
This method implementation (at present) assumes the following:
  (1) i,j are local identifiers, and A has been set up a LocalToGlobalMapping IS.
  (2) The insertion mode of A is GridapDistributedPETScWrappers.ADD_VALUES
"""
function Gridap.Algebra.add_entry!(A::GridapDistributedPETScWrappers.Vec{Float64},
                                   v,
                                   i::Integer)
  GridapDistributedPETScWrappers.set_values_local!(A,
                          GridapDistributedPETScWrappers.PetscInt[i-1],
                          Float64[v])
end
