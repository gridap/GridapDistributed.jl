module MPITests

using Test
using MPI

TESTCASE = get(ENV, "TESTCASE", "all")
@info "Running MPI tests with TESTCASE=$TESTCASE"

if TESTCASE ∈ ("all", "mpi-geometry")
  @time @testset "Geometry" begin include("GeometryTests.jl") end
  @time @testset "CellData" begin include("CellDataTests.jl") end
end

if TESTCASE ∈ ("all", "mpi-fespaces")
  @time @testset "FESpaces"         begin include("FESpacesTests.jl") end
  @time @testset "MultiField"       begin include("MultiFieldTests.jl") end
  @time @testset "ZeroMeanFESpaces" begin include("ZeroMeanFESpacesTests.jl") end
  @time @testset "PeriodicBCs"      begin include("PeriodicBCsTests.jl") end
  @time @testset "ConstantFESpaces" begin include("ConstantFESpacesTests.jl") end
end

if TESTCASE ∈ ("all", "mpi-physics")
  @time @testset "Poisson"            begin include("PoissonTests.jl") end
  @time @testset "PLaplacian"         begin include("PLaplacianTests.jl") end
  @time @testset "SurfaceCoupling"    begin include("SurfaceCouplingTests.jl") end
  @time @testset "StokesOpenBoundary" begin include("StokesOpenBoundaryTests.jl") end
  @time @testset "StokesHdivDG"       begin include("StokesHdivDGTests.jl") end
  @time @testset "HcurlProjection"    begin include("HcurlProjectionTests.jl") end
end

if TESTCASE ∈ ("all", "mpi-transient")
  @time @testset "TransientDistributedCellField" begin
    include("TransientDistributedCellFieldTests.jl")
  end
  @time @testset "TransientMultiFieldDistributedCellField" begin
    include("TransientMultiFieldDistributedCellFieldTests.jl")
  end
  @time @testset "HeatEquation" begin include("HeatEquationTests.jl") end
end

if TESTCASE ∈ ("all", "mpi-adaptivity")
  @time @testset "AdaptivityTests"             begin include("AdaptivityTests.jl") end
  @time @testset "AdaptivityCartesianTests"    begin include("AdaptivityCartesianTests.jl") end
  @time @testset "AdaptivityMultiFieldTests"   begin include("AdaptivityMultiFieldTests.jl") end
  @time @testset "AdaptivityUnstructuredTests" begin include("AdaptivityUnstructuredTests.jl") end
  @time @testset "PolytopalCoarsening"         begin include("PolytopalCoarseningTests.jl") end
end

if TESTCASE ∈ ("all", "mpi-misc")
  @time @testset "BlockSparseMatrixAssemblers" begin
    include("BlockSparseMatrixAssemblersTests.jl")
  end
  @time @testset "BlockPartitionedArrays" begin include("BlockPartitionedArraysTests.jl") end
  @time @testset "Visualization"          begin include("VisualizationTests.jl") end
  @time @testset "Autodiff"               begin include("AutodiffTests.jl") end
  @time @testset "MacroDiscreteModels"    begin include("MacroDiscreteModelsTests.jl") end
end

end # module
