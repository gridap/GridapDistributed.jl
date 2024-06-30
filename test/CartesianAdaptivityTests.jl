module CartesianAdaptivityTests
using Test

using Gridap
using Gridap.Geometry
using Gridap.Adaptivity
using Gridap.FESpaces

using MPI
using GridapDistributed
using PartitionedArrays

using GridapDistributed: i_am_in, generate_subparts
using GridapDistributed: find_local_to_local_map
using GridapDistributed: DistributedAdaptedDiscreteModel, redistribute
using GridapDistributed: RedistributeGlue, redistribute_cell_dofs, redistribute_fe_function, redistribute_free_values

function are_equal(a1::MPIArray,a2::MPIArray)
  same = map(a1,a2) do a1,a2
    a1 ≈ a2
  end
  return reduce(&,same,init=true)
end

function are_equal(a1::DebugArray,a2::DebugArray)
  same = map(a1,a2) do a1,a2
    a1 ≈ a2
  end
  return reduce(&,same,init=true)
end

function are_equal(a1::PVector,a2::PVector)
  are_equal(own_values(a1),own_values(a2))
end

function test_redistribution(coarse_ranks, fine_ranks, model, redist_model, redist_glue)
  sol(x) = sum(x)
  reffe  = ReferenceFE(lagrangian,Float64,1)

  if i_am_in(coarse_ranks)
    space = FESpace(model,reffe)
    u = interpolate(sol,space)
    cell_dofs = map(get_cell_dof_values,local_views(u))
    free_values = get_free_dof_values(u)
    dir_values = zero_dirichlet_values(space)
  else
    space = nothing; u = nothing; cell_dofs = nothing; free_values = nothing; dir_values = nothing;
  end

  redist_space = FESpace(redist_model,reffe)
  redist_u = interpolate(sol,redist_space)
  redist_cell_dofs = map(get_cell_dof_values,local_views(redist_u))
  redist_free_values = get_free_dof_values(redist_u)
  redist_dir_values = zero_dirichlet_values(redist_space)

  # Redistribute cell values, both ways
  tmp_cell_dofs = copy(redist_cell_dofs)
  redistribute_cell_dofs(cell_dofs,tmp_cell_dofs,redist_model,redist_glue)
  @test are_equal(redist_cell_dofs,tmp_cell_dofs)

  tmp_cell_dofs = i_am_in(coarse_ranks) ? copy(cell_dofs) : nothing
  redistribute_cell_dofs(redist_cell_dofs,tmp_cell_dofs,model,redist_glue;reverse=true)
  if i_am_in(coarse_ranks)
    @test are_equal(cell_dofs,tmp_cell_dofs)
  end

  # Redistribute free values, both ways
  tmp_free_values = copy(redist_free_values)
  redistribute_free_values(tmp_free_values,redist_space,free_values,dir_values,space,redist_model,redist_glue)
  @test are_equal(redist_free_values,tmp_free_values)

  tmp_free_values = i_am_in(coarse_ranks) ? copy(free_values) : nothing
  redistribute_free_values(tmp_free_values,space,redist_free_values,redist_dir_values,redist_space,model,redist_glue;reverse=true)
  if i_am_in(coarse_ranks)
    @test are_equal(free_values,tmp_free_values)
  end

  return true
end

function test_adaptivity(ranks,cmodel,fmodel,glue)
  if i_am_in(ranks)
    sol(x) = sum(x)
    reffe  = ReferenceFE(lagrangian,Float64,1)
    amodel = fmodel

    Ωf  = Triangulation(amodel)
    dΩf = Measure(Ωf,2)
    Vf  = FESpace(amodel,reffe)
    Uf  = TrialFESpace(Vf)
    uh_fine = interpolate(sol,Vf)

    Ωc  = Triangulation(cmodel)
    dΩc = Measure(Ωc,2)
    Vc  = FESpace(cmodel,reffe)
    Uc  = TrialFESpace(Vc)
    uh_coarse = interpolate(sol,Vc)

    dΩcf = Measure(Ωc,Ωf,2)

    # Coarse to Fine projection
    af(u,v) = ∫(u⋅v)*dΩf
    lf(v) = ∫(uh_coarse*v)*dΩf
    op = AffineFEOperator(af,lf,Uf,Vf)
    uh_coarse_to_fine = solve(op)

    eh  = uh_fine - uh_coarse_to_fine
    @test sum(∫(eh⋅eh)*dΩf) < 1e-8

    # Fine to Coarse projection
    ac(u,v) = ∫(u⋅v)*dΩc
    lc(v) = ∫(uh_fine*v)*dΩcf
    op = AffineFEOperator(ac,lc,Uc,Vc)
    uh_fine_to_coarse = solve(op)

    eh  = uh_coarse - uh_fine_to_coarse
    @test sum(∫(eh⋅eh)*dΩc) < 1e-8
  end
  return true
end

############################################################################################

function main(distribute,ncells,isperiodic)
  fine_parts = (2,2)
  fine_ranks = distribute(LinearIndices((4,)))

  coarse_parts = (2,1)
  coarse_ranks = generate_subparts(fine_ranks,2)

  # Create models and glues
  if i_am_in(coarse_ranks)
    parent = CartesianDiscreteModel(coarse_ranks,coarse_parts,(0,1,0,1),ncells;isperiodic)
    child  = refine(parent,(2,2))
    coarse_adaptivity_glue = get_adaptivity_glue(child)
  else
    parent = nothing; child  = nothing; coarse_adaptivity_glue = nothing
  end

  redist_parent, redist_glue_parent = redistribute(parent,fine_ranks,fine_parts)
  
  redist_child_1 = refine(redist_parent,(2,2))
  fine_adaptivity_glue = get_adaptivity_glue(redist_child_1)

  redist_child_2, redist_glue_child = redistribute(child,fine_ranks,fine_parts)

  # Tests
  test_redistribution(coarse_ranks,fine_ranks,parent,redist_parent,redist_glue_parent)
  test_redistribution(coarse_ranks,fine_ranks,child,redist_child_2,redist_glue_child)

  test_adaptivity(coarse_ranks,parent,child,coarse_adaptivity_glue)
  test_adaptivity(fine_ranks,redist_parent,redist_child_1,fine_adaptivity_glue)
  return
end

function main(distribute)
  main(distribute,(8,8),(false,false))
  main(distribute,(8,8),(true,false))
  main(distribute,(4,4),(false,true))
  main(distribute,(4,4),(true,true))
end

end # module AdaptivityTests