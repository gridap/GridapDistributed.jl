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

end # module
