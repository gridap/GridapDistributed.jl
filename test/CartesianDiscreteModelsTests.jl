module CartesianDiscreteModelsTests

using Gridap
using GridapDistributed

subdomains = (2,3)
domain = (0,1,0,1)
cells = (10,10)

comm = SequentialCommunicator(subdomains)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

writevtk(model,"model")

end # module
