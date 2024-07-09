using FillArrays
using PartitionedArrays, GridapDistributed
using Gridap
using Gridap.Adaptivity, Gridap.Arrays, Gridap.Geometry

using GridapDistributed: GenericDistributedDiscreteModel

ranks = with_debug() do distribute
  distribute(LinearIndices((2,)))
end

_cmodel = CartesianDiscreteModel(ranks,(2,1),(0,1,0,1),(4,4))
cmodel = UnstructuredDiscreteModel(_cmodel)

fmodel = refine(cmodel)



