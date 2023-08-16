module PeriodicBCsTestsSeq
using PartitionedArrays
using Gridap
using GridapDistributed
using Test

nps = [(1,1),(1,2),(2,1),(2,2)]
domain = (0,1,0,1)
cells = (4,4)
isperiodics = [(false,false),(false,true),(true,false),(true,true)]
reffe = ReferenceFE(lagrangian,Float64,1)
ndofss = [25,20,20,16]
for np in nps
  for (ndofs, isperiodic) in zip(ndofss,isperiodics)
    parts = DebugArray(LinearIndices((prod(np),)))
    model = CartesianDiscreteModel(parts,np,domain,cells;isperiodic=isperiodic)
    V = FESpace(model,reffe)
    @test ndofs == num_free_dofs(V)
  end
end

include("../PeriodicBCsTests.jl")


with_debug() do distribute
  PeriodicBCsTests.main(distribute,(2,2))
end 

with_debug() do distribute
  PeriodicBCsTests.main(distribute,(2,1))
end 

with_debug() do distribute
  PeriodicBCsTests.main(distribute,(1,1))
end 

with_debug() do distribute
  PeriodicBCsTests.main(distribute,(1,2))
end 

with_debug() do distribute 
  PeriodicBCsTests.main(distribute,(2,3))
end

with_debug() do distribute 
  PeriodicBCsTests.main(distribute,(4,1))
end

with_debug() do distribute 
  PeriodicBCsTests.main(distribute,(1,4))
end 

end # module
