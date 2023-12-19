
# SingleField

function ODEs.TransientCellField(f::DistributedCellField,derivatives::Tuple)
  fields = map(local_views(f),map(local_views,derivatives)...) do f, derivatives...
    ODEs.TransientCellField(f,Tuple(derivatives))
  end
  DistributedCellField(fields,get_triangulation(f),f.metadata)
end

function ODEs.time_derivative(f::DistributedCellField)
  fields = map(local_views(f)) do field
    ODEs.time_derivative(field)
  end
  DistributedCellField(fields,get_triangulation(f))
end

# MultiField

function ODEs.TransientCellField(f::DistributedMultiFieldFEFunction,derivatives::Tuple)
  field_fe_fun = map(1:num_fields(f)) do i
    f_i = f[i]
    df_i = Tuple(map(df -> df[i],derivatives))
    ODEs.TransientCellField(f_i,df_i)
  end
  part_fe_fun = map(local_views(f),map(local_views,derivatives)...) do f, derivatives...
    ODEs.TransientCellField(f,Tuple(derivatives))
  end
  free_values = get_free_dof_values(f)
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function ODEs.time_derivative(f::DistributedMultiFieldCellField)
  field_fe_fun = map(ODEs.time_derivative,f.field_fe_fun)
  part_fe_fun  = map(ODEs.time_derivative,f.part_fe_fun)
  DistributedMultiFieldCellField(field_fe_fun,part_fe_fun)
end
