module MPITests

using MPI
using Test

mpidir  = @__DIR__
testdir = joinpath(mpidir, "..")
repodir = joinpath(testdir, "..")

function run_driver(procs, file)
  mpiexec() do cmd
    if MPI.MPI_LIBRARY == "OpenMPI" || (isdefined(MPI, :OpenMPI) && MPI.MPI_LIBRARY == MPI.OpenMPI)
      run(`$cmd -n $procs --oversubscribe $(Base.julia_cmd()) --code-coverage=user --project=$repodir $(joinpath(mpidir, file))`)
    else
      run(`$cmd -n $procs $(Base.julia_cmd()) --code-coverage=user --project=$repodir $(joinpath(mpidir, file))`)
    end
    @test true
  end
end

run_driver(4, "runtests_np4.jl")
run_driver(1, "runtests_np4.jl") # Check that the degenerated case works

end # module
