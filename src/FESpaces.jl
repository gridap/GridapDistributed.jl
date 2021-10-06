#
# Generic FE space related methods

abstract type DistributedFESpace <: GridapType end

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
  allocate_vector(V,num_free_dofs(f))
end

FESpaces.num_free_dofs(f::DistributedFESpace) = length(get_free_dof_ids(f))

function Base.zero(f::DistributedFESpace)
  free_values = zero_free_values(f)
  FEFunction(f,free_values)
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

  # Find the global range of owned dofs
  first_gdof, ngdofsplus1 = xscan(+,reduce,nodofs,init=1)
  ngdofs = ngdofsplus1 - 1

  # Distribute gdofs to owned ones
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
  cell_to_gdofs = map_parts(
    parts,
    ldof_to_gdof,cell_to_ldofs,cell_range.partition) do part,
    ldof_to_gdof,cell_to_ldofs,partition

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
    for cell in partition.oid_to_lid
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      p = ptrs[cell]-1
      for (i,ldof) in enumerate(ldofs)
        if ldof > 0 && ldof_to_gdof[ldof] != gdof
          data[i+p] = ldof_to_gdof[ldof]
        end
      end
    end
    PArrays.Table(data,ptrs)
  end

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

  # Setup dof partition
  dof_partition = map_parts(parts,ldof_to_gdof,ldof_to_part) do part,ldof_to_gdof,ldof_to_part
    IndexSet(part,ldof_to_gdof,ldof_to_part)
  end

  # Setup dof exchanger
  dof_exchanger = Exchanger(dof_partition,neighbors)

  # Setup dof range
  dofs = PRange(ngdofs,dof_partition,dof_exchanger)

  dofs
end

# FEFunction related

struct DistributedFEFunctionData{T<:AbstractVector} <:GridapType
  free_values::T
end

const DistributedFEFunction = DistributedCellField{A,<:DistributedFEFunctionData{T}} where {A,T}

function FESpaces.get_free_dof_values(uh::DistributedFEFunction)
  uh.metadata.free_values
end

# Single field related

struct DistributedSinglefieldFESpace{A,B,C} <: DistributedFESpace
  spaces::A
  gids::B
  vector_type::Type{C}
  function DistributedSinglefieldFESpace(
    spaces::AbstractPData{<:SingleFieldFESpace},
    gids::PRange,
    vector_type::Type{C}) where C
    A = typeof(spaces)
    B = typeof(gids)
    new{A,B,C}(spaces,gids,vector_type)
  end
end

function get_vector_type(fs::DistributedSinglefieldFESpace)
  fs.vector_type
end

function FESpaces.get_free_dof_ids(fs::DistributedFESpace)
  fs.gids
end

function FESpaces.FEFunction(
  f::DistributedSinglefieldFESpace,free_values::AbstractVector)
  local_vals = consistent_local_views(free_values)
  fields = map_parts(FEFunction,f.spaces,local_vals)
  metadata = DistributedFEFunctionData(free_values)
  DistributedCellField(fields,metadata)
end

function FESpaces.EvaluationFunction(
  f::DistributedSinglefieldFESpace,free_values::AbstractVector)
  local_vals = consistent_local_views(free_values)
  fields = map_parts(EvaluationFunction,f.spaces,local_vals)
  metadata = DistributedFEFunctionData(free_values)
  DistributedCellField(fields,metadata)
end

function FESpaces.get_fe_basis(f::DistributedSinglefieldFESpace)
  fields = map_parts(get_fe_basis,f.spaces)
  DistributedCellField(fields)
end

function FESpaces.get_trial_fe_basis(f::DistributedSinglefieldFESpace)
  fields = map_parts(get_trial_fe_basis,f.spaces)
  DistributedCellField(fields)
end

function FESpaces.TrialFESpace(f::DistributedSinglefieldFESpace)
  spaces = map_parts(TrialFESpace,f.spaces)
  DistributedSinglefieldFESpace(spaces,f.gids,f.vector_type)
end

function generate_gids(
  model::DistributedDiscreteModel,
  spaces::AbstractPData{<:SingleFieldFESpace})
  cell_to_ldofs = map_parts(get_cell_dof_ids,spaces)
  nldofs = map_parts(num_free_dofs,spaces)
  generate_gids(model.gids,cell_to_lids,nldofs)
end

# Factories

function FESpace(model::DistributedDiscreteModel,reffe; kwargs...)
  spaces = map_parts(mode.models) do m
    FESpace(m,reffe;kwargs...)
  end
  gids =  generate_gids(model,spaces)
  vector_type = get_vector_type(get_part(spaces))
  DistributedSinglefieldFESpace(spaces,gids,vector_type)
end

# Assembly

function FESpaces.collect_cell_matrix(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  a::DistributedDomainContribution)
  map_parts(collect_cell_matrix,trial.spaces,test.spaces,a.contribs)
end

function FESpaces.collect_cell_vector(
  test::DistributedFESpace, a::DistributedDomainContribution)
  map_parts(collect_cell_vector,test.spaces,a.contribs)
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  biform::DistributedDomainContribution,
  liform::DistributedDomainContribution)
  map_parts(collect_cell_matrix_and_vector,
    trial.spaces,test.spaces,biform.contribs,liform.contribs)
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  biform::DistributedDomainContribution,
  liform::DistributedDomainContribution,
  uhd::DistributedFEFunction)
  map_parts(collect_cell_matrix_and_vector,
    trial.spaces,test.spaces,biform.contribs,liform.contribs,uhd.fields)
end

function FESpaces.collect_cell_vector(
  test::DistributedFESpace,l::Number)
  map_parts(test.spaces) do s
    collect_cell_vector(s,l)
  end
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  mat::DistributedDomainContribution,
  l::Number)
  map_parts(trial.spaces,test.spaces,mat.contribs) do u,v,m
    collect_cell_matrix_and_vector(u,v,m,l)
  end
end

function FESpaces.collect_cell_matrix_and_vector(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  mat::DistributedDomainContribution,
  l::Number,
  uhd::DistributedFEFunction)
  map_parts(trial.spaces,test.spaces,mat.contribs,uhd.fields) do u,v,m,f
    collect_cell_matrix_and_vector(u,v,m,l,f)
  end
end

struct DistributedSparseMatrixAssembler{A,B,C,D,E,F} <: SparseMatrixAssembler
  assems::A
  matrix_builder::B
  vector_builder::C
  rows::D
  cols::E
  strategy::F
end

FESpaces.get_rows(a::DistributedSparseMatrixAssembler) = a.rows
FESpaces.get_cols(a::DistributedSparseMatrixAssembler) = a.cols
FESpaces.get_matrix_builder(a::DistributedSparseMatrixAssembler) = a.matrix_builder
FESpaces.get_vector_builder(a::DistributedSparseMatrixAssembler) = a.vector_builder
FESpaces.get_assembly_strategy(a::DistributedSparseMatrixAssembler) = a.strategy

function FESpaces.symbolic_loop_matrix!(A,a::DistributedSparseMatrixAssembler,matdata)
  map_parts(symbolic_loop_matrix!,local_views(A),a.assems,matdata)
end

function FESpaces.numeric_loop_matrix!(A,a::DistributedSparseMatrixAssembler,matdata)
  map_parts(numeric_loop_matrix!,local_views(A),a.assems,matdata)
end

function FESpaces.symbolic_loop_vector!(b,a::DistributedSparseMatrixAssembler,vecdata)
  map_parts(symbolic_loop_vector!,local_views(b),a.assems,vecdata)
end

function FESpaces.numeric_loop_vector!(b,a::DistributedSparseMatrixAssembler,vecdata)
  map_parts(numeric_loop_vector!,local_views(b),a.assems,vecdata)
end

function FESpaces.symbolic_loop_matrix_and_vector!(A,b,a::DistributedSparseMatrixAssembler,data)
  map_parts(symbolic_loop_matrix_and_vector!,local_views(A),local_views(b),a.assems,data)
end

function FESpaces.numeric_loop_matrix_and_vector!(A,b,a::DistributedSparseMatrixAssembler,data)
  map_parts(numeric_loop_matrix_and_vector!,local_views(A),local_views(b),a.assems,data)
end

