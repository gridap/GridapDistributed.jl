using MPI
include("run_mpi_driver.jl")
file = joinpath(@__DIR__, "drivers", "MacroDiscreteModelsTests.jl")
run_mpi_driver(file; procs=4)
