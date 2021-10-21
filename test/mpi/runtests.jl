module MPITests

using MPI
using Test

#Sysimage
sysimage=nothing
if length(ARGS)==1
   @assert isfile(ARGS[1]) "$(ARGS[1]) must be a valid Julia sysimage file"
   sysimage=ARGS[1]
end 

mpidir = @__DIR__
testdir = joinpath(mpidir,"..")
repodir = joinpath(testdir,"..")
function run_driver(procs,file,sysimage)
  mpiexec() do cmd
    if sysimage!=nothing
       extra_args="-J$(sysimage)"
       run(`$cmd -n $procs $(Base.julia_cmd()) $(extra_args) --project=$repodir $(joinpath(mpidir,file))`)
    else
       run(`$cmd -n $procs $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
    end
    @test true
  end
end

run_driver(4,"runtests_np4.jl",sysimage)
run_driver(1,"runtests_np4.jl",sysimage) # Check that the degenerated case works


end # module
