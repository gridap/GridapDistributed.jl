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

function FESpaces.gather_free_values!(free_values,f::DistributedFESpace,cell_vals)
  map_parts(gather_free_values!, local_views(free_values), local_views(f), local_views(cell_vals))
end

function FESpaces.gather_free_and_dirichlet_values!(free_values,dirichlet_values,f::DistributedFESpace,cell_vals)
  map_parts(gather_free_and_dirichlet_values!, local_views(free_values), local_views(dirichlet_values), local_views(f), local_views(cell_vals))
end

function dof_wise_to_cell_wise!(cell_wise_vector,dof_wise_vector,cell_to_ldofs,cell_prange)
  map_parts(cell_wise_vector,
          dof_wise_vector,
          cell_to_ldofs,
          cell_prange.partition) do cwv,dwv,cell_to_ldofs,partition
    cache  = array_cache(cell_to_ldofs)
    ncells = length(cell_to_ldofs)
    ptrs = cwv.ptrs
    data = cwv.data
    gdof = 0
    for cell in partition.oid_to_lid
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      p = ptrs[cell]-1
      for (i,ldof) in enumerate(ldofs)
        if ldof > 0
          data[i+p] = dwv[ldof]
        end
      end
    end
  end
end

function cell_wise_to_dof_wise!(dof_wise_vector,cell_wise_vector,cell_to_ldofs,cell_range)
  map_parts(dof_wise_vector,
            cell_wise_vector,
            cell_to_ldofs,
            cell_range.partition) do dwv,cwv,cell_to_ldofs,partition

    gdof = 0
    cache = array_cache(cell_to_ldofs)
    for cell in partition.hid_to_lid
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      p = cwv.ptrs[cell]-1
      for (i,ldof) in enumerate(ldofs)
        if ldof > 0
          dwv[ldof] = cwv.data[i+p]
        end
      end
    end
  end
end


function dof_wise_to_cell_wise(dof_wise_vector,cell_to_ldofs,cell_prange)
    cwv=map_parts(dof_wise_vector,cell_to_ldofs,cell_prange.partition) do dwv,cell_to_ldofs,partition
      cache = array_cache(cell_to_ldofs)
      ncells = length(cell_to_ldofs)
      ptrs = Vector{Int32}(undef,ncells+1)
      for cell in 1:ncells
        ldofs = getindex!(cache,cell_to_ldofs,cell)
        ptrs[cell+1] = length(ldofs)
      end
      PArrays.length_to_ptrs!(ptrs)
      ndata = ptrs[end]-1
      data = Vector{Int}(undef,ndata)
      gdof = 0
      PArrays.Table(data,ptrs)
    end
    dof_wise_to_cell_wise!(cwv,dof_wise_vector,cell_to_ldofs,cell_prange)
    cwv
end


function generate_gids(
  cell_range::PRange,
  cell_to_ldofs::AbstractPData{<:AbstractArray},
  nldofs::AbstractPData{<:Integer})

  neighbors = cell_range.exchanger.parts_snd
  ngcells = length(cell_range)

  # Find and count number owned dofs
  ldof_to_part, nodofs = map_parts(
    cell_range.partition,cell_to_ldofs,nldofs) do partition,cell_to_ldofs,nldofs

    ldof_to_part = fill(Int32(0),nldofs)
    cache = array_cache(cell_to_ldofs)
    for cell in 1:length(cell_to_ldofs)
      owner = partition.lid_to_part[cell]
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      for ldof in ldofs
        if ldof>0
          #TODO this simple approach concentrates dofs
          # in the last part and creates inbalances
          ldof_to_part[ldof] = max(owner,ldof_to_part[ldof])
        end
      end
    end
    nodofs = count(p->p==partition.part,ldof_to_part)
    ldof_to_part, nodofs
  end

  cell_ldofs_to_part = dof_wise_to_cell_wise(ldof_to_part,
                                              cell_to_ldofs,
                                              cell_range)
  # Exchange the dof owners
  exchange!(cell_ldofs_to_part,cell_range.exchanger)

  cell_wise_to_dof_wise!(ldof_to_part,
                          cell_ldofs_to_part,
                          cell_to_ldofs,
                          cell_range)


  # Find the global range of owned dofs
  first_gdof, ngdofsplus1 = xscan(+,reduce,nodofs,init=1)
  ngdofs = ngdofsplus1 - 1

  # Distribute gdofs to owned ones
  parts = get_part_ids(nodofs)
  ldof_to_gdof = map_parts(
    parts,first_gdof,ldof_to_part) do part,first_gdof,ldof_to_part

    offset = first_gdof-1
    ldof_to_gdof = Vector{Int}(undef,length(ldof_to_part))
    odof = 0
    gdof = 0
    for (ldof,owner) in enumerate(ldof_to_part)
      if owner == part
        odof += 1
        ldof_to_gdof[ldof] = odof
      else
        ldof_to_gdof[ldof] = gdof
      end
    end
    for (ldof,owner) in enumerate(ldof_to_part)
      if owner == part
        ldof_to_gdof[ldof] += offset
      end
    end
    ldof_to_gdof
  end

  # Create cell-wise global dofs
  cell_to_gdofs = dof_wise_to_cell_wise(ldof_to_gdof,
                                        cell_to_ldofs,
                                        cell_range)

  # Exchange the global dofs
  exchange!(cell_to_gdofs,cell_range.exchanger)


  # Distribute global dof ids also to ghost
  map_parts(
    parts,
    cell_to_ldofs,cell_to_gdofs,ldof_to_gdof,ldof_to_part,cell_range.partition) do part,
    cell_to_ldofs,cell_to_gdofs,ldof_to_gdof,ldof_to_part,partition

    gdof = 0
    cache = array_cache(cell_to_ldofs)
    for cell in partition.hid_to_lid
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      p = cell_to_gdofs.ptrs[cell]-1
      for (i,ldof) in enumerate(ldofs)
        if ldof > 0 && ldof_to_part[ldof] == partition.lid_to_part[cell]
          ldof_to_gdof[ldof] = cell_to_gdofs.data[i+p]
        end
      end
    end
  end

  dof_wise_to_cell_wise!(cell_to_gdofs,ldof_to_gdof,cell_to_ldofs,cell_range)

  exchange!(cell_to_gdofs,cell_range.exchanger)

  cell_wise_to_dof_wise!(ldof_to_gdof,
                          cell_to_gdofs,
                          cell_to_ldofs,
                          cell_range)

  # Setup dof partition
  dof_partition = map_parts(parts,ldof_to_gdof,ldof_to_part) do part,ldof_to_gdof,ldof_to_part
    IndexSet(part,ldof_to_gdof,ldof_to_part)
  end

  # map_parts(parts,dof_partition,cell_to_ldofs,cell_to_gdofs,cell_range.partition) do part, partition, cell_to_ldofs,cell_to_gdofs, cell_range
  #   if (part==3)
  #     println("XXXX $(part)")
  #     println(partition)
  #     println(cell_to_ldofs)
  #     println(cell_to_gdofs)
  #     println(cell_range)
  #   end
  # end

  # Setup dof exchanger
  dof_exchanger = Exchanger(dof_partition,neighbors)

  # Setup dof range
  dofs = PRange(ngdofs,dof_partition,dof_exchanger)

  return dofs
end

# FEFunction related
"""
"""
struct DistributedFEFunctionData{T<:AbstractVector} <:DistributedGridapType
  free_values::T
end

"""
"""
const DistributedSingleFieldFEFunction = DistributedCellField{A,<:DistributedFEFunctionData{T}} where {A,T}

function FESpaces.get_free_dof_values(uh::DistributedSingleFieldFEFunction)
  uh.metadata.free_values
end

# Single field related
"""
"""
struct DistributedSingleFieldFESpace{A,B,C} <: DistributedFESpace
  spaces::A
  gids::B
  vector_type::Type{C}
  function DistributedSingleFieldFESpace(
    spaces::AbstractPData{<:SingleFieldFESpace},
    gids::PRange,
    vector_type::Type{C}) where C
    A = typeof(spaces)
    B = typeof(gids)
    new{A,B,C}(spaces,gids,vector_type)
  end
end

local_views(a::DistributedSingleFieldFESpace) = a.spaces

function FESpaces.get_vector_type(fs::DistributedSingleFieldFESpace)
  fs.vector_type
end

function FESpaces.get_free_dof_ids(fs::DistributedSingleFieldFESpace)
  fs.gids
end

function FESpaces.get_dirichlet_dof_values(U::DistributedSingleFieldFESpace)
  map_parts(get_dirichlet_dof_values,U.spaces)
end

function FESpaces.zero_dirichlet_values(U::DistributedSingleFieldFESpace)
  map_parts(zero_dirichlet_values,U.spaces)
end

function FESpaces.FEFunction(
  f::DistributedSingleFieldFESpace,free_values::AbstractVector,isconsistent=false)
  _EvaluationFunction(FEFunction,f,free_values,isconsistent)
end

function FESpaces.FEFunction(
  f::DistributedSingleFieldFESpace,free_values::AbstractVector,
  dirichlet_values::AbstractPData{<:AbstractVector},isconsistent=false)
  _EvaluationFunction(FEFunction,f,free_values,dirichlet_values,isconsistent)
end

function FESpaces.EvaluationFunction(
  f::DistributedSingleFieldFESpace,free_values::AbstractVector,isconsistent=false)
  _EvaluationFunction(EvaluationFunction,f,free_values,isconsistent)
end

function FESpaces.EvaluationFunction(
  f::DistributedSingleFieldFESpace,free_values::AbstractVector,
  dirichlet_values::AbstractPData{<:AbstractVector},isconsistent=false)
  _EvaluationFunction(EvaluationFunction,f,free_values,dirichlet_values,isconsistent)
end

function _EvaluationFunction(func,
  f::DistributedSingleFieldFESpace,free_values::AbstractVector,isconsistent=false)
  local_vals = consistent_local_views(free_values,f.gids,isconsistent)
  fields = map_parts(func,f.spaces,local_vals)
  metadata = DistributedFEFunctionData(free_values)
  DistributedCellField(fields,metadata)
end

function _EvaluationFunction(func,
  f::DistributedSingleFieldFESpace,free_values::AbstractVector,
  dirichlet_values::AbstractPData{<:AbstractVector},isconsistent=false)
  local_vals = consistent_local_views(free_values,f.gids,isconsistent)
  fields = map_parts(func,f.spaces,local_vals,dirichlet_values)
  metadata = DistributedFEFunctionData(free_values)
  DistributedCellField(fields,metadata)
end

function FESpaces.get_fe_basis(f::DistributedSingleFieldFESpace)
  fields = map_parts(get_fe_basis,f.spaces)
  DistributedCellField(fields)
end

function FESpaces.get_trial_fe_basis(f::DistributedSingleFieldFESpace)
  fields = map_parts(get_trial_fe_basis,f.spaces)
  DistributedCellField(fields)
end

function FESpaces.get_fe_dof_basis(f::DistributedSingleFieldFESpace)
  dofs = map_parts(get_fe_dof_basis,local_views(f))
  DistributedCellDof(dofs)
end

function FESpaces.TrialFESpace(f::DistributedSingleFieldFESpace)
  spaces = map_parts(TrialFESpace,f.spaces)
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FESpaces.TrialFESpace(f::DistributedSingleFieldFESpace,fun)
  spaces = map_parts(f.spaces) do s
    TrialFESpace(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FESpaces.TrialFESpace(fun,f::DistributedSingleFieldFESpace)
  spaces = map_parts(f.spaces) do s
    TrialFESpace(fun,s)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FESpaces.TrialFESpace!(f::DistributedSingleFieldFESpace,fun)
  spaces = map_parts(f.spaces) do s
    TrialFESpace!(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function FESpaces.HomogeneousTrialFESpace(f::DistributedSingleFieldFESpace)
  spaces = map_parts(f.spaces) do s
    HomogeneousTrialFESpace(s)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.vector_type)
end

function generate_gids(
  model::DistributedDiscreteModel{Dc},
  spaces::AbstractPData{<:SingleFieldFESpace}) where Dc
  cell_to_ldofs = map_parts(get_cell_dof_ids,spaces)
  nldofs = map_parts(num_free_dofs,spaces)
  cell_gids = get_cell_gids(model)
  generate_gids(cell_gids,cell_to_ldofs,nldofs)
end

function FESpaces.interpolate(u,f::DistributedSingleFieldFESpace)
  free_values = zero_free_values(f)
  interpolate!(u,free_values,f)
end

function FESpaces.interpolate!(
  u,free_values::AbstractVector,f::DistributedSingleFieldFESpace)
  map_parts(f.spaces,local_views(free_values)) do V,vec
    interpolate!(u,vec,V)
  end
  FEFunction(f,free_values)
end

function FESpaces.interpolate!(
  u::DistributedCellField,free_values::AbstractVector,f::DistributedSingleFieldFESpace)
  map_parts(local_views(u),f.spaces,local_views(free_values)) do ui,V,vec
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
  dirichlet_values::AbstractPData{<:AbstractVector},
  f::DistributedSingleFieldFESpace)
  map_parts(f.spaces,local_views(free_values),dirichlet_values) do V,fvec,dvec
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
  dirichlet_values::AbstractPData{<:AbstractVector},
  f::DistributedSingleFieldFESpace)
  map_parts(f.spaces,local_views(free_values),dirichlet_values) do V,fvec,dvec
    interpolate_everywhere!(u,fvec,dvec,V)
  end
  FEFunction(f,free_values,dirichlet_values)
end

function FESpaces.interpolate_everywhere!(
  u::DistributedCellField, free_values::AbstractVector,
  dirichlet_values::AbstractPData{<:AbstractVector},
  f::DistributedSingleFieldFESpace)
  map_parts(local_views(u),f.spaces,local_views(free_values),dirichlet_values) do ui,V,fvec,dvec
    interpolate_everywhere!(ui,fvec,dvec,V)
  end
  FEFunction(f,free_values,dirichlet_values)
end

# Factories

function FESpaces.FESpace(model::DistributedDiscreteModel,reffe;kwargs...)
  spaces = map_parts(local_views(model)) do m
    FESpace(m,reffe;kwargs...)
  end
  gids =  generate_gids(model,spaces)
  vector_type = _find_vector_type(spaces,gids)
  DistributedSingleFieldFESpace(spaces,gids,vector_type)
end

function FESpaces.FESpace(_trian::DistributedTriangulation,reffe;kwargs...)
  trian = add_ghost_cells(_trian)
  trian_gids = generate_cell_gids(trian)
  spaces = map_parts(trian.trians) do t
    FESpace(t,reffe;kwargs...)
  end
  cell_to_ldofs = map_parts(get_cell_dof_ids,spaces)
  nldofs = map_parts(num_free_dofs,spaces)
  gids = generate_gids(trian_gids,cell_to_ldofs,nldofs)
  vector_type = _find_vector_type(spaces,gids)
  DistributedSingleFieldFESpace(spaces,gids,vector_type)
end

function _find_vector_type(spaces,gids)
  #TODO Now the user can select the local vector type but not the global one
  # new kw-arg global_vector_type ?
  # we use PVector for the moment
  local_vector_type = get_vector_type(get_part(spaces))
  T = eltype(local_vector_type)
  A = typeof(map_parts(i->local_vector_type(undef,0),gids.partition))
  B = typeof(gids)
  vector_type = PVector{T,A,B}
end

# Assembly

function FESpaces.collect_cell_matrix(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  a::DistributedDomainContribution)
  map_parts(
    collect_cell_matrix,
    local_views(trial),
    local_views(test),
    local_views(a))
end

function FESpaces.collect_cell_vector(
  test::DistributedFESpace, a::DistributedDomainContribution)
  map_parts(
    collect_cell_vector,local_views(test),local_views(a))
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  biform::DistributedDomainContribution,
  liform::DistributedDomainContribution)
  map_parts(collect_cell_matrix_and_vector,
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
  map_parts(collect_cell_matrix_and_vector,
    local_views(trial),
    local_views(test),
    local_views(biform),
    local_views(liform),
    local_views(uhd))
end

function FESpaces.collect_cell_vector(
  test::DistributedFESpace,l::Number)
  map_parts(local_views(test)) do s
    collect_cell_vector(s,l)
  end
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  mat::DistributedDomainContribution,
  l::Number)
  map_parts(
    local_views(trial),local_views(test),local_views(mat)) do u,v,m
    collect_cell_matrix_and_vector(u,v,m,l)
  end
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  mat::DistributedDomainContribution,
  l::Number,
  uhd)
  map_parts(
    local_views(trial),local_views(test),local_views(mat),local_views(uhd)) do u,v,m,f
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
  rows::E
  cols::F
end

local_views(a::DistributedSparseMatrixAssembler) = a.assems

FESpaces.get_rows(a::DistributedSparseMatrixAssembler) = a.rows
FESpaces.get_cols(a::DistributedSparseMatrixAssembler) = a.cols
FESpaces.get_matrix_builder(a::DistributedSparseMatrixAssembler) = a.matrix_builder
FESpaces.get_vector_builder(a::DistributedSparseMatrixAssembler) = a.vector_builder
FESpaces.get_assembly_strategy(a::DistributedSparseMatrixAssembler) = a.strategy

function FESpaces.symbolic_loop_matrix!(A,a::DistributedSparseMatrixAssembler,matdata)
  map_parts(symbolic_loop_matrix!,local_views(A,a.rows,a.cols),a.assems,matdata)
end

function FESpaces.numeric_loop_matrix!(A,a::DistributedSparseMatrixAssembler,matdata)
  map_parts(numeric_loop_matrix!,local_views(A,a.rows,a.cols),a.assems,matdata)
end

function FESpaces.symbolic_loop_vector!(b,a::DistributedSparseMatrixAssembler,vecdata)
  map_parts(symbolic_loop_vector!,local_views(b,a.rows),a.assems,vecdata)
end

function FESpaces.numeric_loop_vector!(b,a::DistributedSparseMatrixAssembler,vecdata)
  map_parts(numeric_loop_vector!,local_views(b,a.rows),a.assems,vecdata)
end

function FESpaces.symbolic_loop_matrix_and_vector!(A,b,a::DistributedSparseMatrixAssembler,data)
  map_parts(symbolic_loop_matrix_and_vector!,local_views(A,a.rows,a.cols),local_views(b,a.rows),a.assems,data)
end

function FESpaces.numeric_loop_matrix_and_vector!(A,b,a::DistributedSparseMatrixAssembler,data)
  map_parts(numeric_loop_matrix_and_vector!,local_views(A,a.rows,a.cols),local_views(b,a.rows),a.assems,data)
end

# Parallel Assembly strategies

function local_assembly_strategy(::SubAssembledRows,rows,cols)
  DefaultAssemblyStrategy()
end

# When using this one, make sure that you also loop over ghost cells.
# This is at your own risk.
function local_assembly_strategy(::FullyAssembledRows,rows,cols)
  rows_lid_to_ohid = rows.lid_to_ohid
  GenericAssemblyStrategy(
    identity,
    identity,
    row->rows_lid_to_ohid[row]>0,
    col->true)
end

# Assembler high level constructors
function FESpaces.SparseMatrixAssembler(
  local_mat_type,
  local_vec_type,
  trial::DistributedFESpace,
  test::DistributedFESpace,
  par_strategy=SubAssembledRows())

  Tv = local_vec_type
  T = eltype(Tv)
  Tm = local_mat_type
  cols = trial.gids.partition
  rows = test.gids.partition
  assems = map_parts(local_views(test),local_views(trial),rows,cols) do v,u,rows,cols
    local_strategy = local_assembly_strategy(par_strategy,rows,cols)
    SparseMatrixAssembler(Tm,Tv,u,v,local_strategy)
  end
  matrix_builder = PSparseMatrixBuilderCOO(Tm,par_strategy)
  vector_builder = PVectorBuilder(Tv,par_strategy)
  rows = get_free_dof_ids(test)
  cols = get_free_dof_ids(trial)
  DistributedSparseMatrixAssembler(par_strategy,assems,matrix_builder,vector_builder,rows,cols)
end

function FESpaces.SparseMatrixAssembler(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  par_strategy=SubAssembledRows())
  Tv = get_vector_type(get_part(local_views(trial)))
  T = eltype(Tv)
  Tm = SparseMatrixCSC{T,Int}
  SparseMatrixAssembler(Tm,Tv,trial,test,par_strategy)
end
