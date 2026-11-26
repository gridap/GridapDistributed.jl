module GridapDistributedTests

const GROUPS = ("geometry", "fespaces", "physics", "transient", "adaptivity", "misc")
const TESTCASES = ("all", ("$prefix-$group" for prefix in ("seq", "mpi") for group in GROUPS)...)

TESTCASE = get(ENV, "TESTCASE", "all")

TESTCASE ∈ TESTCASES || error(
  "Unknown TESTCASE=$(TESTCASE), expected one of: $(join(TESTCASES, ", "))."
)

if startswith(TESTCASE, "seq") || TESTCASE == "all"
  include("sequential/runtests.jl")
end

if startswith(TESTCASE, "mpi") || TESTCASE == "all"
  include("mpi/runtests.jl")
end

end # module
