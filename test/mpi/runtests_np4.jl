module NP4
# All test running on 4 procs here

using TestApp
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

if ! MPI.Initialized()
  MPI.Init()
end

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

TestApp.GeometryTests.main(parts)
PArrays.toc!(t,"Geometry")

TestApp.CellDataTests.main(parts)
PArrays.toc!(t,"CellData")

TestApp.FESpacesTests.main(parts)
PArrays.toc!(t,"FESpaces")

TestApp.MultiFieldTests.main(parts)
PArrays.toc!(t,"MultiField")

TestApp.PoissonTests.main(parts)
PArrays.toc!(t,"Poisson")

TestApp.PLaplacianTests.main(parts)
PArrays.toc!(t,"PLaplacian")

display(t)

MPI.Finalize()

end #module
