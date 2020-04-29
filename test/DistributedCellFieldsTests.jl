module DistributedCellFieldsTests

using Gridap
using GridapDistributed

comm = SequentialCommunicator()

subdomains = (2,3)
domain = (0,1,0,1)
cells = (10,10)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

trian = Triangulation(model)

degree = 1
quad = CellQuadrature(trian,degree)

q = get_coordinates(quad)

u(x) = x[1]+x[2]

cf = CellField(u,trian)

ux = evaluate(cf,q)

end # module
