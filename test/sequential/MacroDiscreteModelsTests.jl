module MacroDiscreteModelsTestsSeq

using PartitionedArrays

include("../MacroDiscreteModelsTests.jl")
with_debug() do distribute
  MacroDiscreteModelsTests.main(distribute,(2,2);vtk=true)
end

end # module