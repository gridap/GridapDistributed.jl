# To be executed as
# mpirun -np 4 julia --project=. test/CartesianDiscreteModelsTestsMPI.jl
module CartesianDiscreteModelsTests

using Gridap
using GridapDistributed
using MPI

MPI.Init()

comm = MPICommunicator()

subdomains = (2,2)
domain = (0,1,0,1)
cells = (4,4)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

writevtk(model,"model")

MPI.Finalize()

end # module
