module MPIPETScTests

using Test
using MPI
using GridapDistributed
using GridapDistributedPETScWrappers

nprocs_str = get(ENV, "JULIA_GRIDAPDISTRIBUTED_TEST_NPROCS","")
nprocs = nprocs_str == "" ? clamp(Sys.CPU_THREADS, 2, 4) : parse(Int, nprocs_str)
#mpiexec_args = Base.shell_split("--allow-run-as-root --tag-output") #Base.shell_split(get(ENV, "JULIA_MPIEXEC_TEST_ARGS", ""))
testdir = @__DIR__
istest(f) = endswith(f, ".jl") && startswith(f, "MPI")
testfiles = sort(filter(istest, readdir(testdir)))
@time @testset "$f" for f in testfiles
  MPI.mpiexec() do cmd
     println("$(f)")
     #cmd = `$cmd $mpiexec_args`
     np = nprocs
     if f in ["MPIPETScDistributedVectorsTests.jl","MPIPETScDistributedIndexSetsTests.jl"]
       np = 2
     elseif f in ["MPIPETScDistributedPoissonTests.jl","MPIPETScDistributedPoissonDGTests.jl","MPIPETScDistributedPLaplacianTests.jl","MPIPETScDistributedStokesTests.jl"]
       np = 4
     end
     cmd = `$cmd -n $(np) $(Base.julia_cmd()) --project=. $(joinpath(testdir, f))`
     @show cmd
     run(cmd)
     @test true
  end
end

end # module
