module SequentialTests

using Test

@time @testset "DiscreteModels" begin include("DiscreteModelsTests.jl") end
@time @testset "Triangulations" begin include("TriangulationsTests.jl") end

end # module
