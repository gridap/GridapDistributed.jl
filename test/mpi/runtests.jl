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

@time @testset "Geometry" begin run_driver(4,"GeometryTests.jl")  end
@time @testset "CellData" begin run_driver(4,"CellDataTests.jl")  end
@time @testset "FESpaces" begin run_driver(4,"FESpacesTests.jl")  end
@time @testset "MultiField" begin run_driver(4,"MultiField")  end
@time @testset "Poisson" begin run_driver(4,"Poisson")  end
@time @testset "PLaplacian" begin run_driver(4,"PLaplacian")  end

end # module
