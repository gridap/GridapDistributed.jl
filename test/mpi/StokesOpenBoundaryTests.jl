using MPI
include("run_mpi_driver.jl")
file = joinpath(@__DIR__, "drivers", "StokesOpenBoundaryTests.jl")
run_mpi_driver(file; procs=4)
run_mpi_driver(file; procs=1)
