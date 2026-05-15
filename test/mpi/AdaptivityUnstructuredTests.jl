using MPI
include("run_mpi_driver.jl")
file = joinpath(@__DIR__, "drivers", "AdaptivityUnstructuredTests.jl")
run_mpi_driver(file; procs=4)
