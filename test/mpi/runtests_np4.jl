module NP4
# All test running on 4 procs here

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

if !MPI.Initialized()
  MPI.Init()
end

include("../CellDataTests.jl")
include("../FESpacesTests.jl")
include("../GeometryTests.jl")
include("../MultiFieldTests.jl")
include("../PLaplacianTests.jl")
include("../PoissonTests.jl")
include("../PeriodicBCsTests.jl")
include("../SurfaceCouplingTests.jl")
include("../TransientDistributedCellFieldTests.jl")
include("../TransientMultiFieldDistributedCellFieldTests.jl")
include("../ZeroMeanFESpacesTests.jl")
include("../HeatEquationTests.jl")
include("../StokesOpenBoundaryTests.jl")
include("../AdaptivityTests.jl")
include("../AdaptivityCartesianTests.jl")
include("../AdaptivityUnstructuredTests.jl")
include("../AdaptivityMultiFieldTests.jl")
include("../PolytopalCoarseningTests.jl")
include("../BlockSparseMatrixAssemblersTests.jl")
include("../BlockPartitionedArraysTests.jl")
include("../VisualizationTests.jl")
include("../AutodiffTests.jl")
include("../ConstantFESpacesTests.jl")
include("../MacroDiscreteModelsTests.jl")

include("runtests_np4_body.jl")

if MPI.Comm_size(MPI.COMM_WORLD) == 4
  with_mpi() do distribute
    all_tests(distribute, (2,2))
  end
elseif MPI.Comm_size(MPI.COMM_WORLD) == 1
  with_mpi() do distribute
    all_tests(distribute, (1,1))
  end
else
  MPI.Abort(MPI.COMM_WORLD, 0)
end

end #module
