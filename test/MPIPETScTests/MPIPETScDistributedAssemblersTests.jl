module DistributedAssemblersTests

using Gridap
using Gridap.FESpaces
using GridapDistributed
using GridapDistributedPETScWrappers
using Test
using SparseArrays

const T = Float64
const vector_type = GridapDistributedPETScWrappers.Vec{T}
const matrix_type = GridapDistributedPETScWrappers.Mat{T}

function setup_model(comm)
  domain = (0,1,0,1)
  cells = (4,4)
  subdomains = (2,2)
  model = CartesianDiscreteModel(comm,subdomains,domain,cells)
end

function setup_fe_spaces(model)
  reffe = ReferenceFE(lagrangian,Float64,1)
  V = FESpace(vector_type,model=model,reffe=reffe)
  U = TrialFESpace(V)
  U,V
end

include("../DistributedAssemblersTestsHelpers.jl")

MPIPETScCommunicator() do comm
  model=setup_model(comm)
  U,V=setup_fe_spaces(model)

  das=OwnedAndGhostCellsAssemblyStrategy(V,MapDoFsTypeProcLocal())
  test_assemble(comm,model,U,V,das)
  test_allocate_assemble_add(comm,model,U,V,das)

  das=OwnedCellsAssemblyStrategy(V,MapDoFsTypeProcLocal())
  test_assemble(comm,model,U,V,das)
  test_allocate_assemble_add(comm,model,U,V,das)

end


end # module
