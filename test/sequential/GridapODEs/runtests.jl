module GridapODEsTests

using Test

@time @testset "TransientDistributedCellFieldTests" begin
  include("TransientDistributedCellFieldTests.jl")
end

@time @testset "TransientMultiFieldDistributedCellFieldTests" begin
  include("TransientMultiFieldDistributedCellFieldTests.jl")
end

@time @testset "HeatEquation" begin include("HeatEquationTests.jl") end

@time @testset "StokesOpenBoundary" begin include("StokesOpenBoundaryTests.jl") end

end
