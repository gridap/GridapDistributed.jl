#
# Generic FE space related methods

abstract type DistributedFESpace <: FESpace end

function FESpaces.get_vector_type(fs::DistributedFESpace)
  @abstractmethod
end

function FESpaces.get_free_dof_ids(fs::DistributedFESpace)
  @abstractmethod
end

function FESpaces.FEFunction(f::DistributedFESpace,::AbstractVector)
  @abstractmethod
end

function FESpaces.EvaluationFunction(f::DistributedFESpace,::AbstractVector)
  @abstractmethod
end

function FESpaces.get_fe_basis(f::DistributedFESpace)
  @abstractmethod
end

function FESpaces.get_trial_fe_basis(f::DistributedFESpace)
  @abstractmethod
end

function FESpaces.zero_free_values(f::DistributedFESpace)
  V = get_vector_type(f)
  vec = allocate_vector(V,get_free_dof_ids(f))
  fill!(vec,zero(eltype(vec)))
end

FESpaces.num_free_dofs(f::DistributedFESpace) = length(get_free_dof_ids(f))

function Base.zero(f::DistributedFESpace)
  free_values = zero_free_values(f)
  isconsistent = true
  FEFunction(f,free_values,isconsistent)
end

function FESpaces.get_cell_dof_ids(f::DistributedFESpace)
  map(get_cell_dof_ids,local_views(f))
end

function get_cell_dof_global_ids(f::DistributedFESpace)
  gids = get_free_dof_ids(f)
  map(local_views(f),partition(gids)) do f, gids
    lazy_map(Broadcasting(Reindex(local_to_global(gids))), get_cell_dof_ids(f))
  end
end

function FESpaces.gather_free_values!(free_values,f::DistributedFESpace,cell_vals)
  map(gather_free_values!, local_views(free_values), local_views(f), local_views(cell_vals))
end

function FESpaces.gather_free_and_dirichlet_values!(free_values,dirichlet_values,f::DistributedFESpace,cell_vals)
  map(gather_free_and_dirichlet_values!, local_views(free_values), local_views(dirichlet_values), local_views(f), local_views(cell_vals))
end

function FESpaces.gather_free_and_dirichlet_values(f::DistributedFESpace,cell_vals)
  free_values, dirichlet_values = map(local_views(f),cell_vals) do f, cell_vals
    FESpaces.gather_free_and_dirichlet_values(f,cell_vals)
  end |> tuple_of_arrays
  return free_values, dirichlet_values
end

function dof_wise_to_cell_wise!(cell_wise_vector,dof_wise_vector,cell_to_ldofs,cell_ids)
  map(cell_wise_vector,dof_wise_vector,cell_to_ldofs,cell_ids) do cwv,dwv,cell_to_ldofs,cell_ids
    cache  = array_cache(cell_to_ldofs)
    for cell in cell_ids
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      p = cwv.ptrs[cell]-1
      for (i,ldof) in enumerate(ldofs)
        if ldof > 0
          cwv.data[i+p] = dwv[ldof]
        end
      end
    end
  end
  return cell_wise_vector
end

function cell_wise_to_dof_wise!(dof_wise_vector,cell_wise_vector,cell_to_ldofs,cell_ids)
  map(dof_wise_vector,cell_wise_vector,cell_to_ldofs,cell_ids) do dwv,cwv,cell_to_ldofs,cell_ids
    cache = array_cache(cell_to_ldofs)
    for cell in cell_ids
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      p = cwv.ptrs[cell]-1
      for (i,ldof) in enumerate(ldofs)
        if ldof > 0
          dwv[ldof] = cwv.data[i+p]
        end
      end
    end
  end
  return dof_wise_vector
end

function allocate_cell_wise_vector(T, cell_to_lids)
  map(cell_to_lids) do cell_to_lids
    ptrs = Arrays.generate_ptrs(cell_to_lids)
    data = zeros(T,ptrs[end]-1)
    JaggedArray(data,ptrs)
  end
end

function dof_wise_to_cell_wise(
  dof_wise_vector, cell_to_ldofs, cell_ids; 
  T = eltype(eltype(dof_wise_vector))
)
  cwv = allocate_cell_wise_vector(T,cell_to_ldofs)
  dof_wise_to_cell_wise!(cwv,dof_wise_vector,cell_to_ldofs,cell_ids)
  return cwv
end

function fetch_vector_ghost_values_cache(vector_partition,partition)
  cache = PArrays.p_vector_cache(vector_partition,partition)
  map(reverse,cache)
end

function fetch_vector_ghost_values!(vector_partition,cache)
  assemble!((a,b)->b, vector_partition, cache) 
end

function generate_gids(
  cell_range::PRange,
  cell_to_ldofs::AbstractArray{<:AbstractArray},
  nldofs::AbstractArray{<:Integer}
)
  ranks = linear_indices(partition(cell_range))
  cell_ldofs_to_data = allocate_cell_wise_vector(Int, cell_to_ldofs)
  cache_fetch = fetch_vector_ghost_values_cache(cell_ldofs_to_data,partition(cell_range))

  # Find and count number owned dofs
  ldof_to_owner, nodofs = map(partition(cell_range),cell_to_ldofs,nldofs) do indices,cell_to_ldofs,nldofs
    ldof_to_owner = fill(Int32(0),nldofs)
    cache = array_cache(cell_to_ldofs)
    for (cell, owner) in enumerate(local_to_owner(indices))
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      for ldof in ldofs
        if ldof > 0
          # NOTE: this approach concentrates dofs in the last processor
          ldof_to_owner[ldof] = max(owner,ldof_to_owner[ldof])
        end
      end
    end
    nodofs = count(isequal(part_id(indices)),ldof_to_owner)
    ldof_to_owner, nodofs
  end |> tuple_of_arrays

  # Find the global range of owned dofs
  first_gdof = scan(+,nodofs,type=:exclusive,init=one(eltype(nodofs)))

  # Exchange the dof owners. Cell owner always has correct dof owner.
  cell_ldofs_to_owner = dof_wise_to_cell_wise!(
    cell_ldofs_to_data,ldof_to_owner,cell_to_ldofs,own_to_local(cell_range)
  )
  fetch_vector_ghost_values!(cell_ldofs_to_owner,cache_fetch) |> wait
  cell_wise_to_dof_wise!(
    ldof_to_owner,cell_ldofs_to_owner,cell_to_ldofs,ghost_to_local(cell_range)
  )

  # Fill owned gids
  ldof_to_gdof = map(ranks,first_gdof,ldof_to_owner) do rank,first_gdof,ldof_to_owner
    offset = first_gdof-1
    ldof_to_gdof = zeros(Int,length(ldof_to_owner))
    odof = 0
    for (ldof,owner) in enumerate(ldof_to_owner)
      if owner == rank
        odof += 1
        ldof_to_gdof[ldof] = odof + offset
      end
    end
    ldof_to_gdof
  end

  # Exchange gids
  cell_to_gdofs = dof_wise_to_cell_wise!(
    cell_ldofs_to_data,ldof_to_gdof,cell_to_ldofs,own_to_local(cell_range)
  )
  fetch_vector_ghost_values!(cell_to_gdofs,cache_fetch) |> wait

  # Fill ghost gids with exchanged information
  map(
    cell_to_ldofs,cell_to_gdofs,ldof_to_gdof,ldof_to_owner,partition(cell_range)
  ) do cell_to_ldofs,cell_to_gdofs,ldof_to_gdof,ldof_to_owner,indices
    cache = array_cache(cell_to_ldofs)
    lcell_to_owner = local_to_owner(indices)
    for cell in ghost_to_local(indices)
      p = cell_to_gdofs.ptrs[cell]-1
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      cell_owner = lcell_to_owner[cell]
      for (i,ldof) in enumerate(ldofs)
        if (ldof > 0) && isequal(ldof_to_owner[ldof],cell_owner)
          ldof_to_gdof[ldof] = cell_to_gdofs.data[i+p]
        end
      end
    end
  end

  dof_wise_to_cell_wise!(cell_to_gdofs,ldof_to_gdof,cell_to_ldofs,own_to_local(cell_range))
  fetch_vector_ghost_values!(cell_to_gdofs,cache_fetch) |> wait
  cell_wise_to_dof_wise!(ldof_to_gdof,cell_to_gdofs,cell_to_ldofs,ghost_to_local(cell_range))

  # Setup DoFs LocalIndices
  ngdofs = reduction(+,nodofs,destination=:all,init=zero(eltype(nodofs)))
  local_indices = map(LocalIndices,ngdofs,ranks,ldof_to_gdof,ldof_to_owner)

  return PRange(local_indices)
end

# FEFunction related

"""
"""
struct DistributedFEFunctionData{T<:AbstractVector} <: GridapType
  free_values::T
end

"""
"""
const DistributedSingleFieldFEFunction{A,B,T} = DistributedCellField{A,B,DistributedFEFunctionData{T}}

function FESpaces.get_free_dof_values(uh::DistributedSingleFieldFEFunction)
  uh.metadata.free_values
end

# Single field related
"""
"""
struct DistributedSingleFieldFESpace{A,B,C,D,E} <: DistributedFESpace
  spaces::A
  gids::B
  trian::C
  vector_type::Type{D}
  metadata::E
  function DistributedSingleFieldFESpace(
    spaces::AbstractArray{<:SingleFieldFESpace},
    gids::PRange,
    trian::DistributedTriangulation,
    vector_type::Type{D},
    metadata = nothing
  ) where D
    A = typeof(spaces)
    B = typeof(gids)
    C = typeof(trian)
    E = typeof(metadata)
    new{A,B,C,D,E}(spaces,gids,trian,vector_type,metadata)
  end
end

local_views(a::DistributedSingleFieldFESpace) = a.spaces
CellData.get_triangulation(a::DistributedSingleFieldFESpace) = a.trian

function FESpaces.get_vector_type(fs::DistributedSingleFieldFESpace)
  fs.vector_type
end

function FESpaces.get_free_dof_ids(fs::DistributedSingleFieldFESpace)
  fs.gids
end

function FESpaces.get_dof_value_type(cell_shapefuns::DistributedCellField,cell_dof_basis::DistributedCellDof)
  vt = map(local_views(cell_shapefuns),local_views(cell_dof_basis)) do cell_shapefuns, cell_dof_basis
    FESpaces.get_dof_value_type(cell_shapefuns,cell_dof_basis)
  end
  return PartitionedArrays.getany(vt)
end

function FESpaces.ConstraintStyle(::Type{<:DistributedSingleFieldFESpace{A}}) where A
  return FESpaces.ConstraintStyle(eltype(A))
end

function FESpaces.get_dirichlet_dof_values(U::DistributedSingleFieldFESpace)
  map(get_dirichlet_dof_values,U.spaces)
end

function FESpaces.zero_dirichlet_values(U::DistributedSingleFieldFESpace)
  map(zero_dirichlet_values,U.spaces)
end

function FESpaces.FEFunction(
  f::DistributedSingleFieldFESpace,free_values::AbstractVector,isconsistent=false)
  _EvaluationFunction(FEFunction,f,free_values,isconsistent)
end

function FESpaces.FEFunction(
  f::DistributedSingleFieldFESpace,free_values::AbstractVector,
  dirichlet_values::AbstractArray{<:AbstractVector},isconsistent=false)
  _EvaluationFunction(FEFunction,f,free_values,dirichlet_values,isconsistent)
end

function FESpaces.EvaluationFunction(
  f::DistributedSingleFieldFESpace,free_values::AbstractVector,isconsistent=false)
  _EvaluationFunction(EvaluationFunction,f,free_values,isconsistent)
end

function FESpaces.EvaluationFunction(
  f::DistributedSingleFieldFESpace,free_values::AbstractVector,
  dirichlet_values::AbstractArray{<:AbstractVector},isconsistent=false)
  _EvaluationFunction(EvaluationFunction,f,free_values,dirichlet_values,isconsistent)
end

function _EvaluationFunction(func,
  f::DistributedSingleFieldFESpace,x::AbstractVector,isconsistent=false)
  free_values = change_ghost(x,f.gids,is_consistent=isconsistent,make_consistent=true)
  fields   = map(func,f.spaces,partition(free_values))
  trian    = get_triangulation(f)
  metadata = DistributedFEFunctionData(free_values)
  DistributedCellField(fields,trian,metadata)
end

function _EvaluationFunction(func,
  f::DistributedSingleFieldFESpace,x::AbstractVector,
  dirichlet_values::AbstractArray{<:AbstractVector},isconsistent=false)
  free_values = change_ghost(x,f.gids,is_consistent=isconsistent,make_consistent=true)
  fields   = map(func,f.spaces,partition(free_values),dirichlet_values)
  trian    = get_triangulation(f)
  metadata = DistributedFEFunctionData(free_values)
  DistributedCellField(fields,trian,metadata)
end

function FESpaces.get_fe_basis(f::DistributedSingleFieldFESpace)
  fields = map(get_fe_basis,local_views(f))
  trian  = get_triangulation(f)
  DistributedCellField(fields,trian)
end

function FESpaces.get_trial_fe_basis(f::DistributedSingleFieldFESpace)
  fields = map(get_trial_fe_basis,local_views(f))
  trian  = get_triangulation(f)
  DistributedCellField(fields,trian)
end

function FESpaces.get_fe_dof_basis(f::DistributedSingleFieldFESpace)
  dofs  = map(get_fe_dof_basis,local_views(f))
  trian = get_triangulation(f)
  DistributedCellDof(dofs,trian)
end

function FESpaces.TrialFESpace(f::DistributedSingleFieldFESpace)
  spaces = map(TrialFESpace,f.spaces)
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function FESpaces.TrialFESpace(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialFESpace(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function FESpaces.TrialFESpace(fun,f::DistributedSingleFieldFESpace)
  spaces = map(f.spaces) do s
    TrialFESpace(fun,s)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function FESpaces.TrialFESpace!(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialFESpace!(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function FESpaces.HomogeneousTrialFESpace(f::DistributedSingleFieldFESpace)
  spaces = map(f.spaces) do s
    HomogeneousTrialFESpace(s)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function generate_gids(
  model::DistributedDiscreteModel{Dc},
  spaces::AbstractArray{<:SingleFieldFESpace}
) where Dc
  cell_to_ldofs = map(get_cell_dof_ids,spaces)
  nldofs = map(num_free_dofs,spaces)
  cell_gids = get_cell_gids(model)
  generate_gids(cell_gids,cell_to_ldofs,nldofs)
end

function generate_gids(
  trian::DistributedTriangulation{Dc},
  spaces::AbstractArray{<:SingleFieldFESpace}
) where Dc
  cell_to_ldofs = map(get_cell_dof_ids,spaces)
  nldofs = map(num_free_dofs,spaces)
  cell_gids = generate_cell_gids(trian)
  generate_gids(cell_gids,cell_to_ldofs,nldofs)
end

function FESpaces.interpolate(u,f::DistributedSingleFieldFESpace)
  free_values = zero_free_values(f)
  interpolate!(u,free_values,f)
end

function FESpaces.interpolate!(
  u,free_values::AbstractVector,f::DistributedSingleFieldFESpace)
  map(f.spaces,local_views(free_values)) do V,vec
    interpolate!(u,vec,V)
  end
  FEFunction(f,free_values)
end

function FESpaces.interpolate!(
  u::DistributedCellField,free_values::AbstractVector,f::DistributedSingleFieldFESpace)
  map(local_views(u),f.spaces,local_views(free_values)) do ui,V,vec
    interpolate!(ui,vec,V)
  end
  FEFunction(f,free_values)
end

function FESpaces.interpolate_dirichlet(u, f::DistributedSingleFieldFESpace)
  free_values = zero_free_values(f)
  dirichlet_values = get_dirichlet_dof_values(f)
  interpolate_dirichlet!(u,free_values,dirichlet_values,f)
end

function FESpaces.interpolate_dirichlet!(
  u, free_values::AbstractVector,
  dirichlet_values::AbstractArray{<:AbstractVector},
  f::DistributedSingleFieldFESpace)
  map(f.spaces,local_views(free_values),dirichlet_values) do V,fvec,dvec
    interpolate_dirichlet!(u,fvec,dvec,V)
  end
  FEFunction(f,free_values,dirichlet_values)
end

function FESpaces.interpolate_everywhere(u, f::DistributedSingleFieldFESpace)
  free_values = zero_free_values(f)
  dirichlet_values = get_dirichlet_dof_values(f)
  interpolate_everywhere!(u,free_values,dirichlet_values,f)
end

function FESpaces.interpolate_everywhere!(
  u, free_values::AbstractVector,
  dirichlet_values::AbstractArray{<:AbstractVector},
  f::DistributedSingleFieldFESpace)
  map(f.spaces,local_views(free_values),dirichlet_values) do V,fvec,dvec
    interpolate_everywhere!(u,fvec,dvec,V)
  end
  FEFunction(f,free_values,dirichlet_values)
end

function FESpaces.interpolate_everywhere!(
  u::DistributedCellField, free_values::AbstractVector,
  dirichlet_values::AbstractArray{<:AbstractVector},
  f::DistributedSingleFieldFESpace)
  map(local_views(u),f.spaces,local_views(free_values),dirichlet_values) do ui,V,fvec,dvec
    interpolate_everywhere!(ui,fvec,dvec,V)
  end
  FEFunction(f,free_values,dirichlet_values)
end

# Factories

function FESpaces.FESpace(
  model::DistributedDiscreteModel,reffe;split_own_and_ghost=false,constraint=nothing,kwargs...
)
  spaces = map(local_views(model)) do m
    FESpace(m,reffe;kwargs...)
  end
  gids =  generate_gids(model,spaces)
  trian = DistributedTriangulation(map(get_triangulation,spaces),model)
  vector_type = _find_vector_type(spaces,gids;split_own_and_ghost=split_own_and_ghost)
  space = DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
  return _add_distributed_constraint(space,reffe,constraint)
end

function FESpaces.FESpace(
  _trian::DistributedTriangulation,reffe;split_own_and_ghost=false,constraint=nothing,kwargs...
)
  trian = add_ghost_cells(_trian)
  spaces = map(local_views(trian)) do t
    FESpace(t,reffe;kwargs...)
  end
  gids = generate_gids(trian,spaces)
  vector_type = _find_vector_type(spaces,gids;split_own_and_ghost=split_own_and_ghost)
  space = DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
  return _add_distributed_constraint(space,reffe,constraint)
end

function _find_vector_type(spaces,gids;split_own_and_ghost=false)
  local_vector_type = get_vector_type(PartitionedArrays.getany(spaces))
  Tv = eltype(local_vector_type)
  T  = Vector{Tv}
  if split_own_and_ghost
    T = OwnAndGhostVectors{T}
  end
  if isa(gids,PRange)
    vector_type = typeof(PVector{T}(undef,partition(gids)))
  else # isa(gids,BlockPRange)
    vector_type = typeof(BlockPVector{T}(undef,gids))
  end
  return vector_type
end

# TODO: We would like to avoid this, but I cannot extract the maximal order 
#       from the space itself...
function _add_distributed_constraint(
  F::DistributedFESpace,reffe::ReferenceFE,constraint
)
  order = get_order(reffe)
  _add_distributed_constraint(F,order,constraint)
end

function _add_distributed_constraint(
  F::DistributedFESpace,reffe::Tuple{<:ReferenceFEName,Any,Any},constraint
)
  args = reffe[2]
  order = maximum(args[2])
  _add_distributed_constraint(F,order,constraint)
end

function _add_distributed_constraint(F::DistributedFESpace,order::Integer,constraint)
  if isnothing(constraint)
    V = F
  elseif constraint == :zeromean
    _trian = get_triangulation(F)
    model = get_background_model(_trian)
    trian = remove_ghost_cells(_trian,get_cell_gids(model))
    dΩ = Measure(trian,order)
    V = ZeroMeanFESpace(F,dΩ)
  else
    @unreachable """\n
    The passed option constraint=$constraint is not valid.
    Valid values for constraint: nothing, :zeromean
    """
  end
  V
end

# Assembly

function FESpaces.collect_cell_matrix(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  a::DistributedDomainContribution)
  map(collect_cell_matrix,local_views(trial),local_views(test),local_views(a))
end

function FESpaces.collect_cell_vector(
  test::DistributedFESpace, a::DistributedDomainContribution)
  map(collect_cell_vector,local_views(test),local_views(a))
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  biform::DistributedDomainContribution,
  liform::DistributedDomainContribution)
  map(collect_cell_matrix_and_vector,
    local_views(trial),
    local_views(test),
    local_views(biform),
    local_views(liform))
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  biform::DistributedDomainContribution,
  liform::DistributedDomainContribution,
  uhd)
  map(collect_cell_matrix_and_vector,
    local_views(trial),
    local_views(test),
    local_views(biform),
    local_views(liform),
    local_views(uhd))
end

function FESpaces.collect_cell_vector(
  test::DistributedFESpace,l::Number)
  map(local_views(test)) do s
    collect_cell_vector(s,l)
  end
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  mat::DistributedDomainContribution,
  l::Number)
  map(local_views(trial),local_views(test),local_views(mat)) do u,v,m
    collect_cell_matrix_and_vector(u,v,m,l)
  end
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  mat::DistributedDomainContribution,
  l::Number,
  uhd)
  map(local_views(trial),local_views(test),local_views(mat),local_views(uhd)) do u,v,m,f
    collect_cell_matrix_and_vector(u,v,m,l,f)
  end
end
"""
"""
struct DistributedSparseMatrixAssembler{A,B,C,D,E,F} <: SparseMatrixAssembler
  strategy::A
  assems::B
  matrix_builder::C
  vector_builder::D
  test_dofs_gids_prange::E
  trial_dofs_gids_prange::F
end

local_views(a::DistributedSparseMatrixAssembler) = a.assems

FESpaces.get_rows(a::DistributedSparseMatrixAssembler) = a.test_dofs_gids_prange
FESpaces.get_cols(a::DistributedSparseMatrixAssembler) = a.trial_dofs_gids_prange
FESpaces.get_matrix_builder(a::DistributedSparseMatrixAssembler) = a.matrix_builder
FESpaces.get_vector_builder(a::DistributedSparseMatrixAssembler) = a.vector_builder
FESpaces.get_assembly_strategy(a::DistributedSparseMatrixAssembler) = a.strategy

function FESpaces.symbolic_loop_matrix!(A,a::DistributedSparseMatrixAssembler,matdata)
  rows = get_rows(a)
  cols = get_cols(a)
  map(symbolic_loop_matrix!,local_views(A,rows,cols),local_views(a),matdata)
end

function FESpaces.numeric_loop_matrix!(A,a::DistributedSparseMatrixAssembler,matdata)
  rows = get_rows(a)
  cols = get_cols(a)
  map(numeric_loop_matrix!,local_views(A,rows,cols),local_views(a),matdata)
end

function FESpaces.symbolic_loop_vector!(b,a::DistributedSparseMatrixAssembler,vecdata)
  rows = get_rows(a)
  map(symbolic_loop_vector!,local_views(b,rows),local_views(a),vecdata)
end

function FESpaces.numeric_loop_vector!(b,a::DistributedSparseMatrixAssembler,vecdata)
  rows = get_rows(a)
  map(numeric_loop_vector!,local_views(b,rows),local_views(a),vecdata)
end

function FESpaces.symbolic_loop_matrix_and_vector!(A,b,a::DistributedSparseMatrixAssembler,data)
  rows = get_rows(a)
  cols = get_cols(a)
  Aviews=local_views(A,rows,cols)
  bviews=local_views(b,rows)
  aviews=local_views(a)
  map(symbolic_loop_matrix_and_vector!,Aviews,bviews,aviews,data)  
end

function FESpaces.numeric_loop_matrix_and_vector!(A,b,a::DistributedSparseMatrixAssembler,data)
  rows = get_rows(a)
  cols = get_cols(a)
  map(numeric_loop_matrix_and_vector!,local_views(A,rows,cols),local_views(b,rows),local_views(a),data)
end

# Parallel Assembly strategies

function local_assembly_strategy(::SubAssembledRows,rows,cols)
  DefaultAssemblyStrategy()
end

# When using this one, make sure that you also loop over ghost cells.
# This is at your own risk.
function local_assembly_strategy(::FullyAssembledRows,rows,cols)
  rows_local_to_ghost = local_to_ghost(rows)
  GenericAssemblyStrategy(
    identity,
    identity,
    row->rows_local_to_ghost[row]==0,
    col->true
  )
end

# Assembler high level constructors
function FESpaces.SparseMatrixAssembler(
  local_mat_type,
  local_vec_type,
  rows::PRange,
  cols::PRange,
  par_strategy=SubAssembledRows()
)
  assems = map(partition(rows),partition(cols)) do rows,cols
    local_strategy = local_assembly_strategy(par_strategy,rows,cols)
    FESpaces.GenericSparseMatrixAssembler(
      SparseMatrixBuilder(local_mat_type),
      ArrayBuilder(local_vec_type),
      Base.OneTo(length(rows)),
      Base.OneTo(length(cols)),
      local_strategy
    )
  end
  mat_builder = PSparseMatrixBuilderCOO(local_mat_type,par_strategy)
  vec_builder = PVectorBuilder(local_vec_type,par_strategy)
  return DistributedSparseMatrixAssembler(par_strategy,assems,mat_builder,vec_builder,rows,cols)
end

function FESpaces.SparseMatrixAssembler(
  local_mat_type,
  local_vec_type,
  trial::DistributedFESpace,
  test::DistributedFESpace,
  par_strategy=SubAssembledRows()
)
  rows = get_free_dof_ids(test)
  cols = get_free_dof_ids(trial)
  SparseMatrixAssembler(local_mat_type,local_vec_type,rows,cols,par_strategy)
end

function FESpaces.SparseMatrixAssembler(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  par_strategy=SubAssembledRows()
)
  Tv = PartitionedArrays.getany(map(get_vector_type,local_views(trial)))
  T  = eltype(Tv)
  Tm = SparseMatrixCSC{T,Int}
  SparseMatrixAssembler(Tm,Tv,trial,test,par_strategy)
end

# ZeroMean FESpace
struct DistributedZeroMeanCache{A,B}
  dvol::A
  vol::B
end

const DistributedZeroMeanFESpace{A,B,C,D,E,F} = DistributedSingleFieldFESpace{A,B,C,D,DistributedZeroMeanCache{E,F}}

function FESpaces.FESpaceWithConstantFixed(
  space::DistributedSingleFieldFESpace, 
  gid_to_fix::Int = num_free_dofs(space)
)
  # Find the gid within the processors
  gids = get_free_dof_ids(space)
  lid_to_fix = map(partition(gids)) do gids
    Int(global_to_local(gids)[gid_to_fix]) # returns 0 if not found in the processor
  end

  # Create local spaces
  spaces = map(local_views(space),lid_to_fix) do lspace, lid_to_fix
    fix_constant = !iszero(lid_to_fix)
    FESpaceWithConstantFixed(lspace,fix_constant,lid_to_fix)
  end

  trian = get_triangulation(space)
  gids  = generate_gids(trian,spaces)
  vector_type = _find_vector_type(spaces,gids)
  return DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
end

function FESpaces.ZeroMeanFESpace(space::DistributedSingleFieldFESpace,dΩ::DistributedMeasure)
  # Create underlying space
  _space = FESpaceWithConstantFixed(space,num_free_dofs(space))

  # Setup volume integration
  _vol, dvol = map(local_views(space),local_views(dΩ)) do lspace, dΩ
    dvol = assemble_vector(v -> ∫(v)dΩ, lspace)
    vol  = sum(dvol)
    return vol, dvol
  end |> tuple_of_arrays
  vol  = reduce(+,_vol,init=zero(eltype(vol)))
  metadata = DistributedZeroMeanCache(dvol,vol)

  return DistributedSingleFieldFESpace(
    _space.spaces,_space.gids,_space.trian,_space.vector_type,metadata
  )
end

function FESpaces.FEFunction(
  f::DistributedZeroMeanFESpace,
  free_values::AbstractVector,
  isconsistent=false
)
  dirichlet_values = get_dirichlet_dof_values(f)
  FEFunction(f,free_values,dirichlet_values,isconsistent)
end

function FESpaces.FEFunction(
  f::DistributedZeroMeanFESpace,
  free_values::AbstractVector,
  dirichlet_values::AbstractArray{<:AbstractVector},
  isconsistent=false
)
  free_values = change_ghost(free_values,f.gids,is_consistent=isconsistent,make_consistent=true)
  
  c = _compute_new_distributed_fixedval(
    f,free_values,dirichlet_values
  )
  fv = free_values .+ c # TODO: Do we need to copy, or can we just modify? 
  dv = map(dirichlet_values) do dv
    dv .+ c
  end
  
  fields = map(FEFunction,f.spaces,partition(fv),dv)
  trian = get_triangulation(f)
  metadata = DistributedFEFunctionData(fv)
  DistributedCellField(fields,trian,metadata)
end

# This is required, otherwise we end up calling `FEFunction` with a fixed value of zero, 
# which does not properly interpolate the function provided. 
# With this change, we are interpolating in the unconstrained space and then
# substracting the mean.
function FESpaces.interpolate!(u,free_values::AbstractVector,f::DistributedZeroMeanFESpace)
  dirichlet_values = get_dirichlet_dof_values(f)
  interpolate_everywhere!(u,free_values,dirichlet_values,f)
end
function FESpaces.interpolate!(u::DistributedCellField,free_values::AbstractVector,f::DistributedZeroMeanFESpace)
  dirichlet_values = get_dirichlet_dof_values(f)
  interpolate_everywhere!(u,free_values,dirichlet_values,f)
end

function _compute_new_distributed_fixedval(
  f::DistributedZeroMeanFESpace,fv,dv
)
  dvol = f.metadata.dvol
  vol  = f.metadata.vol
  
  c_i = map(local_views(f),partition(fv),dv,dvol) do space,fv,dv,dvol
    if isa(FESpaces.ConstantApproach(space),FESpaces.FixConstant)
      lid_to_fix = space.dof_to_fix
      c = FESpaces._compute_new_fixedval(fv,dv,dvol,vol,lid_to_fix)
    else
      c = - dot(fv,dvol)/vol
    end
    c
  end
  c = reduce(+,c_i,init=zero(eltype(c_i)))
  return c
end

"""
    ConstantFESpace(
      model::DistributedDiscreteModel; 
      constraint_type=:global, 
      kwargs...
    )

Distributed equivalent to `ConstantFESpace(model;kwargs...)`.

With `constraint_type=:global`, a single dof is shared by all processors.
This creates a global constraint, which is NOT scalable in parallel. Use at your own peril. 

With `constraint_type=:local`, a single dof is owned by each processor and shared with no one else.
This space is locally-constant in each processor, and therefore scalable (but not equivalent
to its serial counterpart). 
"""
function FESpaces.ConstantFESpace(
  model::DistributedDiscreteModel;
  constraint_type=:global,kwargs...
)
  @assert constraint_type ∈ [:global,:local]
  if constraint_type == :global
    msg = "ConstantFESpace is NOT scalable in parallel. For testing purposes only."
    @warn msg
  end

  spaces = map(local_views(model)) do model
    ConstantFESpace(model;kwargs...)
  end

  # Single dof, owned by processor 1 (ghost for all other processors)
  nranks = length(spaces)
  cell_gids = get_cell_gids(model)
  indices = map(partition(cell_gids)) do cell_indices
    me = part_id(cell_indices)
    if constraint_type == :global
      LocalIndices(1,me,Int[1],Int32[1])
    else
      LocalIndices(nranks,me,Int[me],Int32[me])
    end
  end
  gids = PRange(indices)

  trian = DistributedTriangulation(map(get_triangulation,spaces),model)
  vector_type = _find_vector_type(spaces,gids)
  return DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
end
