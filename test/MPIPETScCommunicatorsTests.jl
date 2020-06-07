using MPI
using GridapDistributed

comm = MPIPETScCommunicator()
println("$(MPI.Comm_rank(MPI.COMM_WORLD))")
println("I am processor $(get_part(comm)) out of $(num_parts(comm))")
println("Am I ($(get_part(comm))) master? -> $(i_am_master(comm))")
