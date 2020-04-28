module DistributedTriangulationsTests

using Gridap
using GridapDistributed

comm = SequentialCommunicator()

subdomains = (2,3)
domain = (0,1,0,1)
cells = (10,10)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

trian = Triangulation(model)

btrian = BoundaryTriangulation(model,"boundary")

# TODO fix in Gridap
#strian = SkeletonTriangulation(model,"interior")
strian = SkeletonTriangulation(model)

writevtk(trian,"trian")
writevtk(btrian,"btrian")
writevtk(strian,"strian")

end # module
