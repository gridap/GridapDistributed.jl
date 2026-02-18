
# Distributed FESpace commons

function Arrays.evaluate!(transient_space::DistributedFESpace, space::DistributedFESpace, t::Real)
  map(local_views(transient_space),local_views(space)) do transient_space, space
    Arrays.evaluate!(transient_space,space,t)
  end
  return transient_space
end

# SingleField FESpace

const DistributedTransientTrialFESpace = DistributedSingleFieldFESpace{<:AbstractArray{<:ODEs.TransientTrialFESpace}}

function ODEs.TransientTrialFESpace(space::DistributedSingleFieldFESpace,transient_dirichlet::Union{Function, AbstractVector{<:Function}})
  spaces = map(local_views(space)) do space
    ODEs.TransientTrialFESpace(space,transient_dirichlet)
  end
  gids  = get_free_dof_ids(space)
  trian = get_triangulation(space)
  vector_type = get_vector_type(space)
  DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
end

function ODEs.TransientTrialFESpace(space::DistributedSingleFieldFESpace)
  spaces = map(local_views(space)) do space
    ODEs.TransientTrialFESpace(space)
  end
  gids  = get_free_dof_ids(space)
  trian = get_triangulation(space)
  vector_type = get_vector_type(space)
  DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
end

function ODEs.allocate_space(space::DistributedTransientTrialFESpace)
  spaces = map(local_views(space)) do space
    ODEs.allocate_space(space)
  end
  gids  = get_free_dof_ids(space)
  trian = get_triangulation(space)
  vector_type = get_vector_type(space)
  DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
end

function ODEs.time_derivative(space::DistributedSingleFieldFESpace)
  spaces = map(ODEs.time_derivative,local_views(space))
  gids   = get_free_dof_ids(space)
  trian  = get_triangulation(space)
  vector_type = get_vector_type(space)
  DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
end

for T in [:Real,:Nothing]
  @eval begin
    function Arrays.evaluate(space::DistributedTransientTrialFESpace, t::$T)
      spaces = map(local_views(space)) do space
        Arrays.evaluate(space,t)
      end
      gids  = get_free_dof_ids(space)
      trian = get_triangulation(space)
      vector_type = get_vector_type(space)
      DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
    end
  end
end

# SingleField CellField

const DistributedTransientSingleFieldCellField = DistributedCellField{<:AbstractArray{<:ODEs.TransientSingleFieldCellField}}

function ODEs.TransientCellField(f::DistributedCellField,derivatives::Tuple)
  fields = map(local_views(f),map(local_views,derivatives)...) do f, derivatives...
    ODEs.TransientCellField(f,Tuple(derivatives))
  end
  DistributedCellField(fields,get_triangulation(f),f.metadata)
end

function ODEs.TransientCellField(f::DistributedCellField,derivatives::AbstractArray)
  fields = map(local_views(f),local_views(derivatives)) do f, derivatives
    ODEs.TransientCellField(f,derivatives)
  end
  DistributedCellField(fields,get_triangulation(f),f.metadata)
end

function ODEs.time_derivative(f::DistributedTransientSingleFieldCellField)
  fields = map(local_views(f)) do field
    ODEs.time_derivative(field)
  end
  DistributedCellField(fields,get_triangulation(f))
end

function ODEs.get_cellfield(f::DistributedTransientSingleFieldCellField)
  cellfields = map(local_views(f)) do field
    ODEs.get_cellfield(field)
  end
  DistributedCellField(cellfields,get_triangulation(f))
end

function ODEs.get_derivative(f::DistributedTransientSingleFieldCellField, k::Int)
  derivatives = map(local_views(f)) do field
    ODEs.get_derivative(field, k)
  end
  DistributedCellField(derivatives,get_triangulation(f))
end

function ODEs.get_derivatives(f::DistributedTransientSingleFieldCellField)
  derivatives = map(local_views(f)) do field
    ODEs.get_derivatives(field)
  end
end

# MultiField FESpace

function ODEs.has_transient(space::DistributedMultiFieldFESpace)
  getany(map(ODEs.has_transient,local_views(space)))
end

function ODEs.allocate_space(space::DistributedMultiFieldFESpace)
  if !ODEs.has_transient(space)
    return space
  end
  field_fe_space = map(ODEs.allocate_space,space.field_fe_space)
  style = MultiFieldStyle(space)
  spaces = to_parray_of_arrays(map(local_views,field_fe_space))
  part_fe_spaces = map(s -> MultiFieldFESpace(s;style),spaces)
  gids   = get_free_dof_ids(space)
  vector_type = get_vector_type(space)
  DistributedMultiFieldFESpace(field_fe_space,part_fe_spaces,gids,vector_type)
end

function ODEs.time_derivative(space::DistributedMultiFieldFESpace)
  if !ODEs.has_transient(space)
    return space
  end
  field_fe_space = map(ODEs.time_derivative,space.field_fe_space)
  style = MultiFieldStyle(space)
  spaces = to_parray_of_arrays(map(local_views,field_fe_space))
  part_fe_spaces = map(s -> MultiFieldFESpace(s;style),spaces)
  gids   = get_free_dof_ids(space)
  vector_type = get_vector_type(space)
  DistributedMultiFieldFESpace(field_fe_space,part_fe_spaces,gids,vector_type)
end

for T in [:Real,:Nothing]
  @eval begin
    function Arrays.evaluate(space::DistributedMultiFieldFESpace, t::$T)
      if !ODEs.has_transient(space)
        return space
      end
      field_fe_space = map(s->Arrays.evaluate(s,t),space.field_fe_space)
      style = MultiFieldStyle(space)
      spaces = to_parray_of_arrays(map(local_views,field_fe_space))
      part_fe_spaces = map(s -> MultiFieldFESpace(s;style),spaces)
      gids = get_free_dof_ids(space)
      vector_type = get_vector_type(space)
      DistributedMultiFieldFESpace(field_fe_space,part_fe_spaces,gids,vector_type)
    end
  end
end

# MultiField CellField

const DistributedTransientMultiFieldCellField{A} = 
  DistributedMultiFieldCellField{A,<:AbstractArray{<:ODEs.TransientMultiFieldCellField}}

function ODEs.TransientCellField(f::DistributedMultiFieldCellField,derivatives::Tuple)
  field_fe_fun = map(1:num_fields(f)) do i
    f_i = f[i]
    df_i = Tuple(map(df -> df[i],derivatives))
    ODEs.TransientCellField(f_i,df_i)
  end
  fields = to_parray_of_arrays(map(local_views,field_fe_fun))
  part_fe_fun = map(ODEs.TransientMultiFieldCellField,fields)
  DistributedMultiFieldCellField(field_fe_fun,part_fe_fun,f.metadata)
end

function ODEs.TransientCellField(f::DistributedMultiFieldCellField,derivatives::AbstractArray)
  transient_fields = []
  for i in 1:num_fields(f)
    f_i = f[i]
    df_i = map(local_views(derivatives)) do derivatives
      Tuple(map(df -> df[i],derivatives))
    end
    println(typeof(ODEs.TransientCellField(f_i,df_i)))
    push!(transient_fields, ODEs.TransientCellField(f_i,df_i))
  end
  TransientMultiFieldCellField(transient_fields)
end

function ODEs.TransientMultiFieldCellField(fields::AbstractVector{<:ODEs.TransientSingleFieldCellField})
  cellfield = MultiFieldCellField(map(f -> f.cellfield,fields))
  n_derivatives = length(first(fields).derivatives)
  @check all(map(f -> length(f.derivatives) == n_derivatives,fields))
  derivatives = Tuple(map(i -> MultiFieldCellField(map(f -> f.derivatives[i],fields)),1:n_derivatives))
  TransientMultiFieldCellField(cellfield,derivatives,fields)
end

function ODEs.time_derivative(f::DistributedTransientMultiFieldCellField)
  field_fe_fun = map(ODEs.time_derivative,f.field_fe_fun)
  fields = to_parray_of_arrays(map(local_views,field_fe_fun))
  part_fe_fun = map(ODEs.TransientMultiFieldCellField,fields)
  DistributedMultiFieldCellField(field_fe_fun,part_fe_fun)
end

function ODEs.get_cellfield(f::DistributedTransientMultiFieldCellField)
  field_cellfield = map(ODEs.get_cellfield,f.field_fe_fun)
  cellfields = to_parray_of_arrays(map(local_views,field_cellfield))
  part_cellfields = map(MultiFieldCellField,cellfields)
  DistributedMultiFieldCellField(field_cellfield,part_cellfields)
end

function ODEs.get_derivative(f::DistributedTransientMultiFieldCellField, k::Int)
  field_derivative = map(df -> ODEs.get_derivative(df, k), f.field_fe_fun)
  derivatives = to_parray_of_arrays(map(local_views,field_derivative))
  part_derivatives = map(MultiFieldCellField,derivatives)
  DistributedMultiFieldCellField(field_derivative,part_derivatives)
end

function ODEs.get_derivatives(f::DistributedTransientMultiFieldCellField)
  derivatives = map(local_views(f)) do field
    ODEs.get_derivatives(field)
  end
end
