module CartesianDiscreteModelsTests

using Gridap
using GridapDistributed

comm = SequentialCommunicator()

subdomains = (2,3)
domain = (0,1,0,1)
cells = (10,10)
# I think there is a bug with the numbef of cells being created, should I expect
# 10x10 per subdomain, I get 2x2, I don't see why...
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

writevtk(model,"model")

end # module
