module PullbackTestsSeq
using PartitionedArrays

include("../PullbackTests.jl")
with_debug() do distribute
  PullbackTests.main(distribute,2)
end

end # module