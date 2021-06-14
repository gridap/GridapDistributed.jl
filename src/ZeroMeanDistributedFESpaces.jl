struct ZeroMeanDistributedFESpace{V} <: DistributedFESpace
   vector_type :: Type{V}
   spaces      :: DistributedData{<:FESpace}
   gids        :: DistributedIndexSet
   vol_i       :: DistributedData{Vector{Float64}}
   vol         :: DistributedData{Float64}
 end

function get_vector_type(a::ZeroMeanDistributedFESpace)
   a.vector_type
end

 #  Constructors
function Gridap.TrialFESpace(V::ZeroMeanDistributedFESpace,args...)
  spaces = DistributedData(V.spaces) do part, space
    TrialFESpace(space,args...)
  end
  ZeroMeanDistributedFESpace(get_vector_type(V),spaces,V.gids,V.vol_i,V.vol)
end

# U = TrialFESpace(f.space)
# ZeroMeanFESpace(U,f.vol_i,f.vol,f.constraint_style)

function Gridap.FESpaces.FEFunction(dV::ZeroMeanDistributedFESpace,x)
  dfree_vals = x[dV.gids]
  funs = DistributedData(dV.spaces,dfree_vals) do part, V, free_vals
    FEFunction(V,free_vals)
  end
  dfuns=_generate_zero_mean_funs(dV,funs)
  DistributedFEFunction(dfuns,x,dV)
end

function Gridap.FESpaces.EvaluationFunction(dV::ZeroMeanDistributedFESpace,x)
  dfree_vals = x[dV.gids]
  funs = DistributedData(dV.spaces,dfree_vals) do part, V, free_vals
    Gridap.FESpaces.EvaluationFunction(V,free_vals)
  end
  dfuns=_generate_zero_mean_funs(dV,funs)
  DistributedFEFunction(dfuns,x,dV)
end

function _generate_zero_mean_funs(dV::ZeroMeanDistributedFESpace, funs)
  dpartial_sums_fixed_val=
  DistributedData(dV.spaces, funs, dV.vol_i, dV.vol) do part, V, fun, vol_i, vol
      if (constant_fixed(V))
        fv=get_free_values(fun)
        dv=get_dirichlet_values(fun)
        c=Gridap.FESpaces._compute_new_fixedval(fv,
                                                dv,
                                                vol_i,
                                                vol,
                                                V.dof_to_fix)
      else
        fv=get_free_values(fun)
        c=-dot(fv,vol_i)/vol
      end
      c
  end

  partial_sums_fixed_val=gather(dpartial_sums_fixed_val)
  fixed_val=sum(partial_sums_fixed_val)
  comm=get_comm(dV)
  dfixed_val=scatter_value(comm,fixed_val)

  dfuns = DistributedData(dV.spaces, funs, dfixed_val) do part, V, fun, fixed_val
       free_values=get_free_values(fun)
       fv=apply(+,free_values, Fill(fixed_val,length(free_values)))
       if (constant_fixed(V))
         dirichlet_values=get_dirichlet_values(fun)
         dv = dirichlet_values .+ fixed_val
         return FEFunction(V,fv,dv)
       else
         return FEFunction(V,fv)
       end
  end
end

function constant_fixed(V::FESpaceWithConstantFixed{Gridap.FESpaces.FixConstant})
  true
end

function constant_fixed(V::FESpaceWithConstantFixed)
   false
end


function ZeroMeanDistributedFESpace(::Type{V};
                                    model::DistributedDiscreteModel,
                                    reffe,
                                    kwargs...) where V

  function init_local_spaces(part,model)
    lspace = FESpace(model,reffe,kwargs...)
  end

  comm = get_comm(model)
  spaces = DistributedData(init_local_spaces,comm,model.models)

  dof_lid_to_fix = _compute_dof_lid_to_fix(model,spaces)

  function init_local_spaces_with_dof_removed(part,lspace,dof_lid_to_fix)
    Gridap.FESpaces.FESpaceWithConstantFixed(
      lspace, dof_lid_to_fix != -1, dof_lid_to_fix)
  end

  spaces_dof_removed = DistributedData(init_local_spaces_with_dof_removed,
                                       comm,
                                       spaces,
                                       dof_lid_to_fix)

  # TO-DO: order=Gridap.FESpaces._get_kwarg(:order,kwargs)
  order=1
  dvol_i, dvol = _setup_vols(model,spaces,order)
  gids=_compute_distributed_index_set(model, spaces_dof_removed)
  ZeroMeanDistributedFESpace(V,spaces_dof_removed,gids,dvol_i,dvol)
end


function _setup_vols(model,spaces,order)
  comm = get_comm(model)
  dvol_i_and_vol = DistributedData(model,spaces) do part, (model,gids), lspace
    trian = Triangulation(model)
    owned_trian = remove_ghost_cells(trian, part, gids)
    owned_quad = CellQuadrature(owned_trian, order)
    Gridap.FESpaces._setup_vols(lspace,owned_trian,owned_quad)
  end
  dvol_i=DistributedData(dvol_i_and_vol) do part, vol_i_and_vol
    vol_i_and_vol[1]
  end
  dvol_partial_sums=DistributedData(dvol_i_and_vol) do part, vol_i_and_vol
    vol_i_and_vol[2]
  end
  partial_sums=gather(dvol_partial_sums)
  if (i_am_master(comm))
     vol=sum(partial_sums)
  end
  dvol=scatter_value(comm,vol)
  (dvol_i,dvol)
end

function _compute_dof_lid_to_fix(model,spaces)
  dof_lids_candidates=DistributedData(model.gids,spaces) do part, cell_gids, lspace
   n_free_dofs = num_free_dofs(lspace)
   lid_to_n_local_minus_ghost=zeros(Int32,n_free_dofs)
   cell_dofs=get_cell_dof_ids(lspace)
   cell_dofs_cache = array_cache(cell_dofs)
   for cell in 1:length(cell_dofs)
    current_cell_dofs = getindex!(cell_dofs_cache,cell_dofs,cell)
    is_local = (cell_gids.lid_to_owner[cell] == part)
    for lid in current_cell_dofs
     if (lid>0)
       if (is_local)
         lid_to_n_local_minus_ghost[lid] += 1
       else
         lid_to_n_local_minus_ghost[lid] -= 1
       end
     end
    end
   end
   min_lid_only_local_cells=
    findfirst(x->(x>0),lid_to_n_local_minus_ghost)
   min_lid_only_local_cells==nothing ? -1 : min_lid_only_local_cells
  end

  comm = get_comm(model)
  part_dof_lids_candidates = gather(dof_lids_candidates)
  if i_am_master(comm)
    first_proc = findfirst(x->(x!=-1), part_dof_lids_candidates)
    for proc=first_proc+1:length(part_dof_lids_candidates)
      part_dof_lids_candidates[proc]=-1
    end
  end

  dof_lid_to_fix = scatter(comm,part_dof_lids_candidates)
end
