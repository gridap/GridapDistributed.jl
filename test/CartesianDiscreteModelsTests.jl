module CartesianDiscreteModelsTests

using Gridap
using GridapDistributed

comm = SequentialCommunicator()

subdomains = (2,3)
domain = (0,1,0,1)
cells = (10,10)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

writevtk(model,"model")

end # module
