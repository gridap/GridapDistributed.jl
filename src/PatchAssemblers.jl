
# PatchTopology

struct DistributedPatchTopology{Dc,Dp,A}
  topos :: AbstractArray{A}
  function DistributedPatchTopology(
    topos::AbstractArray{<:Geometry.PatchTopology{Dc,Dp}}
  ) where {Dc,Dp}
    A = eltype(topos)
    new{Dc,Dp,A}(topos)
  end
end

local_views(ptopo::DistributedPatchTopology) = ptopo.topos

function Geometry.PatchTopology(
  topo::DistributedGridTopology,patch_cells::AbstractArray{<:Table},metadata=map(x -> nothing, patch_cells)
)
  topos = map(Geometry.PatchTopology,local_views(topo),patch_cells,metadata)
  DistributedPatchTopology(topos)
end

function Geometry.PatchTopology(
  ::Type{ReferenceFE{Df}},model::DistributedDiscreteModel;
  labels = get_face_labeling(model), tags = nothing
) where Df
  Dc = num_cell_dims(model)
  topo = get_grid_topology(model)
  face_gids = get_face_gids(model,Df)
  patch_cells, metadata = map(local_views(topo),local_views(labels),partition(face_gids)) do topo, labels, indices
    patch_cells = get_faces(topo,Df,Dc)
    patch_roots = own_to_local(indices)
    if !isnothing(tags)
      mask = Geometry.get_face_mask(labels,tags,Df)
      patch_roots = filter(p -> mask[p], patch_roots)
    end
    metadata = Geometry.StarPatchMetadata(Int8(Df),patch_roots)
    return patch_cells[patch_roots], metadata
  end |> tuple_of_arrays
  return Geometry.PatchTopology(topo,patch_cells,metadata)
end

# PatchTriangulation

function Geometry.PatchTriangulation(model::DistributedDiscreteModel,ptopo::DistributedPatchTopology;kwargs...)
  trians = map(local_views(model),local_views(ptopo)) do model, ptopo
    Geometry.PatchTriangulation(model,ptopo;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

function Geometry.PatchBoundaryTriangulation(model::DistributedDiscreteModel,ptopo::DistributedPatchTopology;kwargs...)
  trians = map(local_views(model),local_views(ptopo)) do model, ptopo
    Geometry.PatchBoundaryTriangulation(model,ptopo;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

function Geometry.PatchSkeletonTriangulation(model::DistributedDiscreteModel,ptopo::DistributedPatchTopology;kwargs...)
  trians = map(local_views(model),local_views(ptopo)) do model, ptopo
    Geometry.PatchSkeletonTriangulation(model,ptopo;kwargs...)
  end
  DistributedTriangulation(trians,model)
end

# PatchFESpace

function FESpaces.PatchFESpace(
  space::DistributedSingleFieldFESpace,ptopo::DistributedPatchTopology
)
  spaces = map(FESpaces.PatchFESpace,local_views(space),local_views(ptopo))
  return DistributedSingleFieldFESpace(
    spaces, space.gids, space.trian, space.vector_type, space.metadata
  )
end

function FESpaces.PatchFESpace(
  space::DistributedMultiFieldFESpace,ptopo::DistributedPatchTopology
)
  sf_space = map(s -> FESpaces.PatchFESpace(s,ptopo), space)
  return MultiFieldFESpace(sf_space; style = MultiFieldStyle(space))
end

# SpaceWithoutBCs

function FESpaces.FESpaceWithoutBCs(space::DistributedSingleFieldFESpace)
  trian = get_triangulation(space)
  spaces = map(FESpaces.FESpaceWithoutBCs,local_views(space))
  gids = generate_gids(trian,spaces)
  return DistributedSingleFieldFESpace(
    spaces, gids, trian, space.vector_type, space.metadata
  )
end

function FESpaces.FESpaceWithoutBCs(space::DistributedMultiFieldFESpace)
  sf_space = map(FESpaces.FESpaceWithoutBCs,space)
  return MultiFieldFESpace(sf_space; style = MultiFieldStyle(space))
end

# LocalOperators

struct DistributedLocalOperator
  ops :: AbstractArray{<:LocalOperator}
  model :: DistributedDiscreteModel
end

local_views(P::DistributedLocalOperator) = P.ops

(P::DistributedLocalOperator)(u) = evaluate(P,u)

function Arrays.evaluate!(
  cache,k::DistributedLocalOperator,v::DistributedCellField
)
  fields = map(evaluate,local_views(k),local_views(v))
  trians = map(get_triangulation,fields)
  trian = DistributedTriangulation(trians,k.model)
  return DistributedCellField(fields,trian)
end

function Arrays.evaluate!(
  cache,k::DistributedLocalOperator,v::DistributedMultiFieldCellField
)
  n_fields = num_fields(v)
  mf_fields = map(evaluate,local_views(k),local_views(v))
  sf_fields = map(1:n_fields) do field 
    sf_fields = map(f -> f[field], mf_fields)
    trians = map(get_triangulation,sf_fields)
    trian = DistributedTriangulation(trians,k.model)
    DistributedCellField(sf_fields,trian)
  end
  return DistributedMultiFieldCellField(sf_fields, mf_fields)
end

# Patch assembly 

struct DistributedPatchAssembler{A,B} <: Assembler
  assems :: A
  axes :: NTuple{2,PRange{B}}
end

local_views(assem::DistributedPatchAssembler) = assem.assems

function FESpaces.PatchAssembler(
  ptopo::DistributedPatchTopology,trial::DistributedFESpace,test::DistributedFESpace;kwargs...
)
  assems = map(local_views(ptopo),local_views(trial),local_views(test)) do ptopo,trial,test
    FESpaces.PatchAssembler(ptopo,trial,test;kwargs...)
  end
  rows = get_free_dof_ids(trial)
  cols = get_free_dof_ids(test)
  return DistributedPatchAssembler(assems,(rows,cols))
end

for func in (:assemble_matrix,:assemble_vector,:assemble_matrix_and_vector)
  @eval begin
    function FESpaces.$func(assem::DistributedPatchAssembler,celldata)
      map(FESpaces.$func,local_views(assem),celldata)
    end
  end
end

for func in (:collect_patch_cell_matrix,:collect_patch_cell_vector,:collect_patch_cell_matrix_and_vector)
  @eval begin
    function FESpaces.$func(assem::DistributedPatchAssembler,args...)
      local_args = map(local_views,args)
      map(FESpaces.$func,local_views(assem),local_args...)
    end
  end
end

function FESpaces.assemble_matrix(f::Function,a::DistributedPatchAssembler,U::DistributedFESpace,V::DistributedFESpace)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  FESpaces.assemble_matrix(a,FESpaces.collect_patch_cell_matrix(a,U,V,f(u,v)))
end

function FESpaces.assemble_vector(f::Function,a::DistributedPatchAssembler,V::DistributedFESpace)
  v = get_fe_basis(V)
  FESpaces.assemble_vector(a,FESpaces.collect_patch_cell_vector(a,V,f(v)))
end

function FESpaces.assemble_matrix_and_vector(
  f::Function,b::Function,a::DistributedPatchAssembler,U::DistributedFESpace,V::DistributedFESpace
)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  FESpaces.assemble_matrix_and_vector(a,FESpaces.collect_patch_cell_matrix_and_vector(a,U,V,f(u,v),b(v)))
end

function FESpaces.assemble_matrix_and_vector(
  f::Function,b::Function,a::DistributedPatchAssembler,U::DistributedFESpace,V::DistributedFESpace,uhd
)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  FESpaces.assemble_matrix_and_vector(a,FESpaces.collect_patch_cell_matrix_and_vector(a,U,V,f(u,v),b(v),uhd))
end

# StaticCondensationOperator

function MultiField.statically_condensed_assembly(
  retained_assem::DistributedSparseMatrixAssembler,patch_assem,full_matvecs
)
  data = map(local_views(patch_assem),full_matvecs) do patch_assem, full_matvecs
    sc_matvecs = lazy_map(MultiField.StaticCondensationMap(),full_matvecs)
    rows = patch_assem.strategy.array.array[2,2].patch_rows
    cols = patch_assem.strategy.array.array[2,2].patch_cols
    return (([sc_matvecs,],[rows,],[cols,]), ([],[],[]), ([],[]))
  end
  FESpaces.assemble_matrix_and_vector(retained_assem,data)
end

function MultiField.backward_static_condensation!(
  x_eliminated,eliminated_assem::DistributedSparseMatrixAssembler,patch_assem,full_matvecs,x_retained
)
  vecdata = map(local_views(patch_assem),full_matvecs,partition(x_retained)) do patch_assem, full_matvecs, x_retained
    rows_elim = patch_assem.strategy.array.array[1,1].patch_rows
    rows_ret = patch_assem.strategy.array.array[2,2].patch_rows

    patch_x_ret = lazy_map(Broadcasting(Reindex(x_retained)),rows_ret)
    patch_x_elim = lazy_map(MultiField.BackwardStaticCondensationMap(),full_matvecs,patch_x_ret)

    return ([patch_x_elim,],[rows_elim,])
  end
  FESpaces.assemble_vector!(x_eliminated,eliminated_assem,vecdata)
end

# merge_assembly_data

function FESpaces.merge_assembly_matvec_data(data::AbstractArray...)
  map(FESpaces.merge_assembly_matvec_data,data...)
end
