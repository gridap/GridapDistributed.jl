module AdaptivityTests
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
using GridapDistributed: DistributedAdaptedDiscreteModel
using GridapDistributed: RedistributeGlue, redistribute_cell_dofs, redistribute_fe_function, redistribute_free_values

function are_equal(a1::MPIArray,a2::MPIArray)
  same = map(a1,a2) do a1,a2
    a1 ≈ a2
  end
  return reduce(&,same,init=true)
end

function are_equal(a1::PVector,a2::PVector)
  are_equal(own_values(a1),own_values(a2))
end

function DistributedAdaptivityGlue(serial_glue,parent,child)
  glue = map(partition(get_cell_gids(parent)),partition(get_cell_gids(child))) do parent_gids, child_gids
    old_g2l = global_to_local(parent_gids)
    old_l2g = local_to_global(parent_gids)
    new_l2g = local_to_global(child_gids)

    n2o_cell_map  = lazy_map(Reindex(old_g2l),serial_glue.n2o_faces_map[3][new_l2g])
    n2o_faces_map = [Int64[],Int64[],collect(n2o_cell_map)]
    n2o_cell_to_child_id = serial_glue.n2o_cell_to_child_id[new_l2g]
    rrules = serial_glue.refinement_rules[old_l2g]
    AdaptivityGlue(n2o_faces_map,n2o_cell_to_child_id,rrules)
  end
  return glue
end

function get_redistribute_glue(old_parts,new_parts::DebugArray,old_cell_to_part,new_cell_to_part,model,redist_model)
  parts_rcv,parts_snd,lids_rcv,lids_snd,old2new,new2old = 
  map(new_parts,partition(get_cell_gids(redist_model))) do p,new_partition
    old_new_cell_to_part = collect(zip(old_cell_to_part,new_cell_to_part))
    gids_rcv = findall(x->x[1]!=p && x[2]==p, old_new_cell_to_part)
    gids_snd = findall(x->x[1]==p && x[2]!=p, old_new_cell_to_part)

    parts_rcv = unique(old_cell_to_part[gids_rcv])
    parts_snd = unique(new_cell_to_part[gids_snd])

    gids_rcv_by_part = [filter(x -> old_cell_to_part[x] == nbor, gids_rcv) for nbor in parts_rcv]
    gids_snd_by_part = [filter(x -> new_cell_to_part[x] == nbor, gids_snd) for nbor in parts_snd]

    if p ∈ old_parts.items
      old_partition = partition(get_cell_gids(model)).items[p]
      lids_rcv = map(gids -> lazy_map(Reindex(global_to_local(new_partition)),gids),gids_rcv_by_part)
      lids_snd = map(gids -> lazy_map(Reindex(global_to_local(old_partition)),gids),gids_snd_by_part)
      old2new = replace(find_local_to_local_map(old_partition,new_partition), -1 => 0)
      new2old = replace(find_local_to_local_map(new_partition,old_partition), -1 => 0)
    else
      lids_rcv = map(gids -> lazy_map(Reindex(global_to_local(new_partition)),gids),gids_rcv_by_part)
      lids_snd = map(gids -> fill(Int32(0),length(gids)),gids_snd_by_part)
      old2new = Int[]
      new2old = fill(0,length(findall(x -> x == p,new_cell_to_part)))
    end

    return parts_rcv,parts_snd,JaggedArray(lids_rcv),JaggedArray(lids_snd),old2new,new2old
  end |> tuple_of_arrays

  return RedistributeGlue(new_parts,old_parts,parts_rcv,parts_snd,lids_rcv,lids_snd,old2new,new2old)
end

function get_redistribute_glue(old_parts,new_parts::MPIArray,old_cell_to_part,new_cell_to_part,model,redist_model)
  parts_rcv,parts_snd,lids_rcv,lids_snd,old2new,new2old = 
  map(new_parts,partition(get_cell_gids(redist_model))) do p,new_partition
    old_new_cell_to_part = collect(zip(old_cell_to_part,new_cell_to_part))
    gids_rcv = findall(x->x[1]!=p && x[2]==p, old_new_cell_to_part)
    gids_snd = findall(x->x[1]==p && x[2]!=p, old_new_cell_to_part)

    parts_rcv = unique(old_cell_to_part[gids_rcv])
    parts_snd = unique(new_cell_to_part[gids_snd])

    gids_rcv_by_part = [filter(x -> old_cell_to_part[x] == nbor, gids_rcv) for nbor in parts_rcv]
    gids_snd_by_part = [filter(x -> new_cell_to_part[x] == nbor, gids_snd) for nbor in parts_snd]

    if i_am_in(old_parts)
      old_partition = PartitionedArrays.getany(partition(get_cell_gids(model)))
      lids_rcv = map(gids -> lazy_map(Reindex(global_to_local(new_partition)),gids),gids_rcv_by_part)
      lids_snd = map(gids -> lazy_map(Reindex(global_to_local(old_partition)),gids),gids_snd_by_part)
      old2new = replace(find_local_to_local_map(old_partition,new_partition), -1 => 0)
      new2old = replace(find_local_to_local_map(new_partition,old_partition), -1 => 0)
    else
      lids_rcv = map(gids -> lazy_map(Reindex(global_to_local(new_partition)),gids),gids_rcv_by_part)
      lids_snd = map(gids -> fill(Int32(0),length(gids)),gids_snd_by_part)
      old2new = Int[]
      new2old = fill(0,length(findall(x -> x == p,new_cell_to_part)))
    end

    return parts_rcv,parts_snd,JaggedArray(lids_rcv),JaggedArray(lids_snd),old2new,new2old
  end |> tuple_of_arrays

  return RedistributeGlue(new_parts,old_parts,parts_rcv,parts_snd,lids_rcv,lids_snd,old2new,new2old)
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
    amodel = DistributedAdaptedDiscreteModel(fmodel,cmodel,glue)

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

function main(distribute)
  fine_ranks = distribute(LinearIndices((4,)))
  coarse_ranks = generate_subparts(fine_ranks,2)

  # Create models and glues 
  serial_parent = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(4,4)))
  serial_child  = refine(serial_parent)
  serial_rglue  = get_adaptivity_glue(serial_child)

  parent_cell_to_part = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]
  child_cell_to_part  = lazy_map(Reindex(parent_cell_to_part),serial_rglue.n2o_faces_map[3])
  if i_am_in(coarse_ranks)
    parent = DiscreteModel(coarse_ranks,serial_parent,parent_cell_to_part)
    child  = DiscreteModel(coarse_ranks,serial_child,child_cell_to_part)
    coarse_adaptivity_glue = DistributedAdaptivityGlue(serial_rglue,parent,child)
  else
    parent = nothing; child  = nothing; coarse_adaptivity_glue = nothing
  end

  redist_parent_cell_to_part = [1,1,2,2,1,1,2,2,3,3,4,4,3,3,4,4]
  redist_parent = DiscreteModel(fine_ranks,serial_parent,redist_parent_cell_to_part)
  redist_glue_parent = get_redistribute_glue(coarse_ranks,fine_ranks,parent_cell_to_part,redist_parent_cell_to_part,parent,redist_parent);

  redist_child_cell_to_part = lazy_map(Reindex(redist_parent_cell_to_part),serial_rglue.n2o_faces_map[3])
  redist_child = DiscreteModel(fine_ranks,serial_child,redist_child_cell_to_part)
  fine_adaptivity_glue = DistributedAdaptivityGlue(serial_rglue,redist_parent,redist_child)
  redist_glue_child = get_redistribute_glue(coarse_ranks,fine_ranks,child_cell_to_part,redist_child_cell_to_part,child,redist_child);

  # Tests
  test_redistribution(coarse_ranks,fine_ranks,parent,redist_parent,redist_glue_parent)
  test_redistribution(coarse_ranks,fine_ranks,child,redist_child,redist_glue_child)

  test_adaptivity(coarse_ranks,parent,child,coarse_adaptivity_glue)
  test_adaptivity(fine_ranks,redist_parent,redist_child,fine_adaptivity_glue)

  return
end

end # module AdaptivityTests