module SequentialTests

using Test

@time @testset "Geometry" begin include("GeometryTests.jl") end

@time @testset "PeriodicBCs" begin include("PeriodicBCsTests.jl") end

@time @testset "CellData" begin include("CellDataTests.jl") end

@time @testset "FESpaces" begin include("FESpacesTests.jl") end

@time @testset "MultiField" begin include("MultiFieldTests.jl") end

@time @testset "Poisson" begin include("PoissonTests.jl") end

@time @testset "PLaplacian" begin include("PLaplacianTests.jl") end

@time @testset "DivConformingTests" begin include("DivConformingTests.jl") end

@time @testset "SurfaceCouplingTests" begin include("SurfaceCouplingTests.jl") end

@time @testset "TransientDistributedCellFieldTests" begin
  include("TransientDistributedCellFieldTests.jl")
end

@time @testset "TransientMultiFieldDistributedCellFieldTests" begin
  include("TransientMultiFieldDistributedCellFieldTests.jl")
end

@time @testset "HeatEquation" begin include("HeatEquationTests.jl") end

@time @testset "StokesOpenBoundary" begin include("StokesOpenBoundaryTests.jl") end

@time @testset "StokesHdivDGTests.jl" begin include("StokesHdivDGTests.jl") end


end # module
