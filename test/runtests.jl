module GridapDistributedTests

TESTCASE = get(ENV, "TESTCASE", "all")

if startswith(TESTCASE, "seq") || TESTCASE == "all"
  include("sequential/runtests.jl")
end

if startswith(TESTCASE, "mpi") || TESTCASE == "all"
  include("mpi/runtests.jl")
end

end # module
