module GeometryTestsSeq
using PartitionedArrays
include("../GeometryTests.jl")

with_debug() do distribute
  GeometryTests.main(distribute,(2,2))
end

with_debug() do distribute
  GeometryTests.main(distribute,(2,2,2))
end

end
