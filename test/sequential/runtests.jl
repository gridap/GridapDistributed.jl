module SequentialTests

using Test

@time @testset "Geometry" begin include("GeometryTests.jl") end

@time @testset "CellData" begin include("CellDataTests.jl") end

@time @testset "FESpaces" begin include("FESpacesTests.jl") end

@time @testset "MultiField" begin include("MultiFieldTests.jl") end

end # module
