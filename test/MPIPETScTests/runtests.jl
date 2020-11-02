module MPIPETScTests

using Test
using MPI
using GridapDistributed
using GridapDistributedPETScWrappers

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--image-file", "-i"
        help = "Path to the image file that one can use in order to accelerate MPIPETSc tests"
        arg_type = String
        default="GridapDistributed.so"
    end
    return parse_args(s)
end

parsed_args = parse_commandline()
image_file_path=parsed_args["image-file"]
image_file_exists=isfile(image_file_path)

nprocs_str = get(ENV, "JULIA_GRIDAPDISTRIBUTED_TEST_NPROCS","")
nprocs = nprocs_str == "" ? clamp(Sys.CPU_THREADS, 2, 4) : parse(Int, nprocs_str)
#mpiexec_args = Base.shell_split("--allow-run-as-root --tag-output") #Base.shell_split(get(ENV, "JULIA_MPIEXEC_TEST_ARGS", ""))
testdir = @__DIR__
istest(f) = endswith(f, ".jl") && startswith(f, "MPI")
testfiles = sort(filter(istest, readdir(testdir)))
@time @testset "$f" for f in testfiles
  MPI.mpiexec() do cmd
     #cmd = `$cmd $mpiexec_args`
     np = nprocs
     extra_args = ""
     if f in ["MPIPETScDistributedVectorsTests.jl","MPIPETScDistributedIndexSetsTests.jl"]
       np = 2
     elseif f in ["MPIPETScDistributedPoissonTests.jl"]
       np = 4
       extra_args = "-s 2 2 -p 4 4"
     elseif f in ["MPIPETScDistributedPoissonDGTests.jl","MPIPETScDistributedPLaplacianTests.jl","MPIPETScDistributedStokesTests.jl"]
       np = 4
     end
     if ! image_file_exists
       cmd = `$cmd -n $(np) $(Base.julia_cmd()) --project=. $(joinpath(testdir, f)) $(split(extra_args))`
     else
      cmd = `$cmd -n $(np) $(Base.julia_cmd()) -J$(image_file_path) --project=. $(joinpath(testdir, f)) $(split(extra_args))`
     end
     @show cmd
     run(cmd)
     @test true
  end
end

end # module
