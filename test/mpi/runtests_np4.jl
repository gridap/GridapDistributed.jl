module NP4
# All test running on 4 procs here

using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

if ! MPI.Initialized()
  MPI.Init()
end

include("../GeometryTests.jl")
include("../CellDataTests.jl")
include("../FESpacesTests.jl")
include("../MultiFieldTests.jl")
include("../PoissonTests.jl")
include("../PLaplacianTests.jl")

if MPI.Comm_size(MPI.COMM_WORLD) == 4
  parts = get_part_ids(mpi,(2,2))
elseif MPI.Comm_size(MPI.COMM_WORLD) == 1
  parts = get_part_ids(mpi,(1,1))
else
  error()
end

display(parts)

t = PArrays.PTimer(parts,verbose=true)
PArrays.tic!(t)

GeometryTests.main(parts)
PArrays.toc!(t,"Geometry")

CellDataTests.main(parts)
PArrays.toc!(t,"CellData")

FESpacesTests.main(parts)
PArrays.toc!(t,"FESpaces")

MultiFieldTests.main(parts)
PArrays.toc!(t,"MultiField")

PoissonTests.main(parts)
PArrays.toc!(t,"Poisson")

PLaplacianTests.main(parts)
PArrays.toc!(t,"PLaplacian")

display(t)

end #module
