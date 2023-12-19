
# Distributed FESpace

function Arrays.evaluate!(transient_space::DistributedFESpace, space::DistributedFESpace, t::Real)
  map(local_values(transient_space),local_views(space)) do transient_space, space
    Arrays.evaluate!(transient_space,space,t)
  end
  return space
end

# SingleField FESpace

const DistributedTransientTrialFESpace = DistributedSingleFieldFESpace{AbstractArray{<:ODEs.AbstractTransientTrialFESpace}}

function ODEs.TransientTrialFESpace(space::DistributedSingleFieldFESpace,args...)
  spaces = map(local_views(space)) do space
    ODEs.TransientTrialFESpace(space,args...)
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

function ODEs.time_derivative(space::DistributedTransientTrialFESpace)
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

const DistributedTransientSingleFieldCellField{A} = DistributedCellField{A,<:AbstractArray{<:ODEs.TransientSingleFieldCellField}}

function ODEs.TransientCellField(f::DistributedCellField,derivatives::Tuple)
  fields = map(local_views(f),map(local_views,derivatives)...) do f, derivatives...
    ODEs.TransientCellField(f,Tuple(derivatives))
  end
  DistributedCellField(fields,get_triangulation(f),f.metadata)
end

function ODEs.time_derivative(f::DistributedTransientSingleFieldCellField)
  fields = map(local_views(f)) do field
    ODEs.time_derivative(field)
  end
  DistributedCellField(fields,get_triangulation(f))
end

# MultiField FESpace

const DistributedTransientMultiFieldFESpace{MS,A} = 
  DistributedMultiFieldFESpace{MS,A,<:AbstractArray{<:ODEs.TransientMultiFieldFESpace}}

function ODEs.TransientMultiFieldFESpace(spaces::Vector{<:DistributedSingleFieldFESpace})
  MultiFieldFESpace(spaces)
end

function ODEs.allocate_space(space::DistributedTransientMultiFieldFESpace{MS}) where MS
  field_fe_spaces = map(ODEs.allocate_space,space.field_fe_spaces)
  spaces = to_parray_of_arrays(map(local_views,field_fe_spaces))
  part_fe_spaces = map(s -> MultiFieldFESpace(s;style=MS()),spaces)
  gids   = get_free_dof_ids(space)
  vector_type = get_vector_type(space)
  DistributedMultiFieldFESpace(field_fe_spaces,part_fe_spaces,gids,vector_type)
end

function ODEs.time_derivative(f::DistributedTransientMultiFieldFESpace{MS}) where MS
  field_fe_spaces = map(ODEs.time_derivative,f.field_fe_spaces)
  spaces = to_parray_of_arrays(map(local_views,field_fe_spaces))
  part_fe_spaces = map(s -> MultiFieldFESpace(s;style=MS()),spaces)
  gids   = get_free_dof_ids(f)
  vector_type = get_vector_type(f)
  DistributedMultiFieldFESpace(field_fe_spaces,part_fe_spaces,gids,vector_type)
end

for T in [:Real,:Nothing]
  @eval begin
    function Arrays.evaluate(space::DistributedTransientMultiFieldFESpace{MS}, t::$T) where MS
      field_fe_spaces = map(s->Arrays.evaluate(s,t),space.field_fe_spaces)
      spaces = to_parray_of_arrays(map(local_views,field_fe_spaces))
      part_fe_spaces = map(s -> MultiFieldFESpace(s;style=MS()),spaces)
      gids = get_free_dof_ids(space)
      vector_type = get_vector_type(space)
      DistributedMultiFieldFESpace(field_fe_spaces,part_fe_spaces,gids,vector_type)
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
  part_fe_fun = map(MultiField.MultiFieldCellField,fields)
  DistributedMultiFieldCellField(field_fe_fun,part_fe_fun,f.metadata)
end

function ODEs.time_derivative(f::DistributedTransientMultiFieldCellField)
  field_fe_fun = map(ODEs.time_derivative,f.field_fe_fun)
  fields = to_parray_of_arrays(map(local_views,field_fe_fun))
  part_fe_fun = map(MultiField.MultiFieldCellField,fields)
  DistributedMultiFieldCellField(field_fe_fun,part_fe_fun)
end
