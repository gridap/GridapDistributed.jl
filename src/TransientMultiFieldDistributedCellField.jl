# Transient MultiField
struct TransientMultiFieldDistributedCellField{A} <: TransientDistributedCellField
  cellfield::A
  derivatives::Tuple
  transient_single_fields::Vector{<:TransientDistributedCellField} # used to iterate
end

local_views(f::TransientMultiFieldDistributedCellField) = local_views(f.cellfield)

# Constructors
function TransientFETools.TransientCellField(multi_field::DistributedMultiFieldFEFunction,derivatives::Tuple)
  transient_single_fields = _to_transient_single_distributed_fields(multi_field,derivatives)
  TransientMultiFieldDistributedCellField(multi_field,derivatives,transient_single_fields)
end

# Get single index
function Base.getindex(f::TransientMultiFieldDistributedCellField,ifield::Integer)
  single_field = f.cellfield[ifield]
  single_derivatives = ()
  for ifield_derivatives in f.derivatives
    single_derivatives = (single_derivatives...,getindex(ifield_derivatives,ifield))
  end
  TransientSingleFieldDistributedCellField(single_field,single_derivatives)
end

# Get multiple indices
function Base.getindex(f::TransientMultiFieldDistributedCellField,indices::Vector{<:Int})
  cellfield = DistributedMultiFieldCellField(f.cellfield[indices],DomainStyle(f.cellfield))
  derivatives = ()
  for derivative in f.derivatives
    derivatives = (derivatives...,DistributedMultiFieldCellField(derivative[indices],DomainStyle(derivative)))
  end
  transient_single_fields = _to_transient_single_distributed_fields(cellfield,derivatives)
  TransientMultiFieldDistributedCellField(cellfield,derivatives,transient_single_fields)
end

function _to_transient_single_distributed_fields(multi_field,derivatives)
  transient_single_fields = TransientDistributedCellField[]
  for ifield in 1:num_fields(multi_field)
    single_field = multi_field[ifield]
    single_derivatives = ()
    for ifield_derivatives in derivatives
      single_derivatives = (single_derivatives...,getindex(ifield_derivatives,ifield))
    end
    transient_single_field = TransientSingleFieldDistributedCellField(single_field,single_derivatives)
    push!(transient_single_fields,transient_single_field)
  end
  transient_single_fields
end

# Iterate functions
Base.iterate(f::TransientMultiFieldDistributedCellField)  = iterate(f.transient_single_fields)
Base.iterate(f::TransientMultiFieldDistributedCellField,state)  = iterate(f.transient_single_fields,state)

# Time derivative
function ∂t(f::TransientMultiFieldDistributedCellField)
  cellfield, derivatives = first_and_tail(f.derivatives)
  transient_single_field_derivatives = TransientDistributedCellField[]
  for transient_single_field in f.transient_single_fields
    push!(transient_single_field_derivatives,∂t(transient_single_field))
  end
  TransientMultiFieldDistributedCellField(cellfield,derivatives,transient_single_field_derivatives)
end
