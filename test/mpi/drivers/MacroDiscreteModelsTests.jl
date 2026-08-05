module MacroDiscreteModelsMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "MacroDiscreteModelsTests.jl"))

with_mpi() do distribute
  MacroDiscreteModelsTests.main(distribute, (2, 2))
end

end # module
