module VisualizationTestsSeq
  include("../VisualizationTests.jl")
  using PartitionedArrays
  with_debug() do distribute
    VisualizationTests.main(distribute,(2,2))
  end
end
