module GridapDistributedTests

using Test

@time @testset "CartesianDiscreteModels" begin include("CartesianDiscreteModelsTests.jl") end

end # module
