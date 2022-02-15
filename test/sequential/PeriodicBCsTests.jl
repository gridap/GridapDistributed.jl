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
    parts = get_part_ids(sequential,np)
    model = CartesianDiscreteModel(parts,domain,cells;isperiodic=isperiodic)
    V = FESpace(model,reffe)
    @test ndofs == num_free_dofs(V)
  end
end

include("../PeriodicBCsTests.jl")
prun(PeriodicBCsTests.main,sequential,(2,2))
prun(PeriodicBCsTests.main,sequential,(2,1))
prun(PeriodicBCsTests.main,sequential,(1,1))
prun(PeriodicBCsTests.main,sequential,(1,2))
prun(PeriodicBCsTests.main,sequential,(2,3))
end # module
