module ConstantFESpacesTestsSeq

using PartitionedArrays
include("../ConstantFESpacesTests.jl")

with_debug() do distribute
  ConstantFESpacesTests.main(distribute,(2,2))
end

end
