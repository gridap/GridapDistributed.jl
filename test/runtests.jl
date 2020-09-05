module GridapDistributedTests

using Test
using MPI
using PETSc

@time @testset "DistributedData" begin include("DistributedDataTests.jl") end

@time @testset "DistributedIndexSets" begin include("DistributedIndexSetsTests.jl") end

@time @testset "DistributedVectors" begin include("DistributedVectorsTests.jl") end

@time @testset "CartesianDiscreteModels" begin include("CartesianDiscreteModelsTests.jl") end

@time @testset "DistributedFESpaces" begin include("DistributedFESpacesTests.jl") end

@time @testset "ZeroMeanDistributedFESpacesTests" begin include("ZeroMeanDistributedFESpacesTests.jl") end

@time @testset "DistributedAssemblers" begin include("DistributedAssemblersTests.jl") end

@time @testset "DistributedPoisson" begin include("DistributedPoissonTests.jl") end

@time @testset "DistributedPoissonDG" begin include("DistributedPoissonDGTests.jl") end

@time @testset "DistributedPLaplacian" begin include("DistributedPLaplacianTests.jl") end

@time @testset "DistributedStokes" begin include("DistributedStokesTests.jl") end


nprocs_str = get(ENV, "JULIA_GRIDAPDISTRIBUTED_TEST_NPROCS","")
nprocs = nprocs_str == "" ? clamp(Sys.CPU_THREADS, 2, 4) : parse(Int, nprocs_str)
mpiexec_args = Base.shell_split("--allow-run-as-root --tag-output") #Base.shell_split(get(ENV, "JULIA_MPIEXEC_TEST_ARGS", ""))
testdir = @__DIR__
istest(f) = endswith(f, ".jl") && startswith(f, "MPI")
testfiles = sort(filter(istest, readdir(testdir)))
@testset "$f" for f in testfiles
  MPI.mpiexec() do cmd
     println("$(f)")
     cmd = `$cmd $mpiexec_args`
     np = nprocs
     if f in ["MPIPETScDistributedVectorsTests.jl","MPIPETScDistributedIndexSetsTests.jl"]
       np = 2
     elseif f in ["MPIPETScDistributedPoissonTests.jl","MPIPETScDistributedPoissonDGTests.jl","MPIPETScDistributedPLaplacianTests.jl","MPIPETScDistributedStokesTests.jl"]
       np = 4
     end
     cmd = `$cmd -n $(np) $(Base.julia_cmd()) $(joinpath(testdir, f))`
     @show cmd
     run(cmd)
     @test true
  end
end

end # module
