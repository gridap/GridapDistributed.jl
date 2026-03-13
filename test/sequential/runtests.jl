module SequentialTests

using Test

TESTCASE = get(ENV, "TESTCASE", "all")

if TESTCASE ∈ ("all", "seq-geometry")
  @time @testset "Geometry" begin include("GeometryTests.jl") end
  @time @testset "CellData" begin include("CellDataTests.jl") end
end

if TESTCASE ∈ ("all", "seq-fespaces")
  @time @testset "FESpaces"         begin include("FESpacesTests.jl") end
  @time @testset "MultiField"       begin include("MultiFieldTests.jl") end
  @time @testset "issue_142"        begin include("issue_142.jl") end
  @time @testset "ZeroMeanFESpaces" begin include("ZeroMeanFESpacesTests.jl") end
  @time @testset "PeriodicBCs"      begin include("PeriodicBCsTests.jl") end
end

if TESTCASE ∈ ("all", "seq-physics")
  @time @testset "Poisson"                     begin include("PoissonTests.jl") end
  @time @testset "PLaplacian"                  begin include("PLaplacianTests.jl") end
  @time @testset "DivAndCurlConformingTests"   begin include("DivAndCurlConformingTests.jl") end
  @time @testset "SurfaceCouplingTests"        begin include("SurfaceCouplingTests.jl") end
  @time @testset "StokesHdivDGTests"           begin include("StokesHdivDGTests.jl") end
  @time @testset "StokesOpenBoundary"          begin include("StokesOpenBoundaryTests.jl") end
end

if TESTCASE ∈ ("all", "seq-transient")
  @time @testset "TransientDistributedCellFieldTests" begin
    include("TransientDistributedCellFieldTests.jl")
  end
  @time @testset "TransientMultiFieldDistributedCellFieldTests" begin
    include("TransientMultiFieldDistributedCellFieldTests.jl")
  end
  @time @testset "HeatEquation" begin include("HeatEquationTests.jl") end
end

if TESTCASE ∈ ("all", "seq-adaptivity")
  @time @testset "AdaptivityTests" begin include("AdaptivityTests.jl") end
end

if TESTCASE ∈ ("all", "seq-misc")
  @time @testset "BlockSparseMatrixAssemblers" begin
    include("BlockSparseMatrixAssemblersTests.jl")
  end
end

end # module
