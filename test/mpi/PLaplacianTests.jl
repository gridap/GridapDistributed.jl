using MPI
include("run_mpi_driver.jl")
file = joinpath(@__DIR__, "drivers", "PLaplacianTests.jl")
run_mpi_driver(file; procs=4)
run_mpi_driver(file; procs=1)
