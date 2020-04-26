module GridapDistributedTests

using Test

@time @testset "CartesianDiscreteModels" begin include("CartesianDiscreteModelsTests.jl") end

@time @testset "DistributedFESpaces" begin include("DistributedFESpacesTests.jl") end

end # module
