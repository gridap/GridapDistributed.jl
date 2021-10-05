module SequentialTests

using Test

@time @testset "Geometry" begin include("GeometryTests.jl") end

@time @testset "CellData" begin include("CellDataTests.jl") end

end # module
