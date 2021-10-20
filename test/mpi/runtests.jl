module MPITests

using MPI
using Test

mpidir = @__DIR__
testdir = joinpath(mpidir,"..")
repodir = joinpath(testdir,"..")

function run_driver(procs,file)
  mpiexec() do cmd
    run(`$cmd -n $procs $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
    @test true
  end
end

run_driver(1,"runtests_np4.jl") # Check that the degenerated case works
run_driver(4,"runtests_np4.jl")


end # module
