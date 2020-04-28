module GridapDistributedTests

using Test

@time @testset "CartesianDiscreteModels" begin include("CartesianDiscreteModelsTests.jl") end

@time @testset "DistributedFESpaces" begin include("DistributedFESpacesTests.jl") end

@time @testset "GloballyAddressableArrays" begin include("GloballyAddressableArraysTests.jl") end

end # module
