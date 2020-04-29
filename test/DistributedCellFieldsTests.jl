module DistributedCellFieldsTests

using Gridap
using GridapDistributed

subdomains = (2,3)
comm = SequentialCommunicator(subdomains)

domain = (0,1,0,1)
cells = (10,10)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

trian = Triangulation(model)

degree = 1
quad = CellQuadrature(trian,degree)

q = get_coordinates(quad)

u(x) = x[1]+x[2]

cf = CellField(u,trian)

writevtk(trian,"cellfield",cellfields=["u"=>cf])

ux = evaluate(cf,q)

end # module
