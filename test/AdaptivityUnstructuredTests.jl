module AdaptivityUnstructuredTests
using Test

using Gridap
using Gridap.Geometry
using Gridap.Adaptivity
using Gridap.FESpaces

using MPI
using GridapDistributed
using PartitionedArrays

using GridapDistributed: i_am_in

function test_adaptivity(ranks,cmodel,fmodel)
  if i_am_in(ranks)
    sol(x) = sum(x)
    order  = 1
    qorder = 2*order
    reffe  = ReferenceFE(lagrangian,Float64,order)
    amodel = fmodel

    Ωf  = Triangulation(amodel)
    dΩf = Measure(Ωf,qorder)
    Vf  = FESpace(amodel,reffe)
    Uf  = TrialFESpace(Vf)
    uh_fine = interpolate(sol,Vf)

    Ωc  = Triangulation(cmodel)
    dΩc = Measure(Ωc,qorder)
    Vc  = FESpace(cmodel,reffe)
    Uc  = TrialFESpace(Vc)
    uh_coarse = interpolate(sol,Vc)

    dΩcf = Measure(Ωc,Ωf,qorder)

    # Coarse to Fine projection
    af(u,v) = ∫(u⋅v)*dΩf
    lf(v) = ∫(uh_coarse*v)*dΩf
    op = AffineFEOperator(af,lf,Uf,Vf)
    uh_coarse_to_fine = solve(op)

    eh = uh_fine - uh_coarse_to_fine
    @test sum(∫(eh⋅eh)*dΩf) < 1e-6

    # Fine to Coarse projection
    ac(u,v) = ∫(u⋅v)*dΩc
    lc(v) = ∫(uh_fine*v)*dΩcf
    op = AffineFEOperator(ac,lc,Uc,Vc)
    uh_fine_to_coarse = solve(op)

    eh = uh_coarse - uh_fine_to_coarse
    @test sum(∫(eh⋅eh)*dΩc) < 1e-6
  end
  return true
end

############################################################################################

function main(distribute,parts,ncells)
  ranks = distribute(LinearIndices((prod(parts),)))

  Dc = length(ncells)
  domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
  parent1 = UnstructuredDiscreteModel(
    CartesianDiscreteModel(ranks,parts,domain,ncells)
  )

  i_am_main(ranks) && println("UnstructuredAdaptivityTests: red_green")
  child1 = refine(parent1, refinement_method = "red_green" )
  test_adaptivity(ranks,parent1,child1)

  i_am_main(ranks) && println("UnstructuredAdaptivityTests: simplexify")
  child2 = refine(parent1, refinement_method = "simplexify" )
  test_adaptivity(ranks,parent1,child2)

  parent2 = simplexify(parent1,positive=true)

  if Dc == 2
    i_am_main(ranks) && println("UnstructuredAdaptivityTests: nvb")
    child3 = refine(parent2, refinement_method = "nvb" )
    test_adaptivity(ranks,parent2,child3)
  end

  i_am_main(ranks) && println("UnstructuredAdaptivityTests: barycentric")
  child4 = refine(parent2, refinement_method = "barycentric" )
  test_adaptivity(ranks,parent2,child4)
end

function main(distribute)
  main(distribute,(2,2),(8,8))
  main(distribute,(2,2,1),(4,4,4))
end

end # module AdaptivityUnstructuredTests
