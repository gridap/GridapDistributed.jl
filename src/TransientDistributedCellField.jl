# Transient Distributed CellField
abstract type TransientDistributedCellField <: DistributedCellDatum  end

# Transient SingleField
struct TransientSingleFieldDistributedCellField{A} <: TransientDistributedCellField
  cellfield::A
  derivatives::Tuple
end

local_views(f::TransientSingleFieldDistributedCellField) = local_views(f.cellfield)

# Constructors
function TransientFETools.TransientCellField(single_field::DistributedSingleFieldFEFunction,derivatives::Tuple)
  TransientSingleFieldDistributedCellField(single_field,derivatives)
end

function TransientFETools.TransientCellField(single_field::DistributedCellField,derivatives::Tuple)
TransientSingleFieldDistributedCellField(single_field,derivatives)
end

# Time derivative
function ∂t(f::TransientDistributedCellField)
  cellfield, derivatives = first_and_tail(f.derivatives)
  TransientCellField(cellfield,derivatives)
end

∂tt(f::TransientDistributedCellField) = ∂t(∂t(f))

# Integration related
function Fields.integrate(f::TransientDistributedCellField,b::DistributedMeasure)
  integrate(f.cellfield,b)
end

# Differential Operations
Fields.gradient(f::TransientDistributedCellField) = gradient(f.cellfield)
Fields.∇∇(f::TransientDistributedCellField) = ∇∇(f.cellfield)

# Unary ops
for op in (:symmetric_part,:inv,:det,:abs,:abs2,:+,:-,:tr,:transpose,:adjoint,:grad2curl,:real,:imag,:conj)
  @eval begin
    ($op)(a::TransientDistributedCellField) = ($op)(a.cellfield)
  end
end

# Binary ops
for op in (:inner,:outer,:double_contraction,:+,:-,:*,:cross,:dot,:/)
  @eval begin
    ($op)(a::TransientDistributedCellField,b::TransientDistributedCellField) = ($op)(a.cellfield,b.cellfield)
    ($op)(a::TransientDistributedCellField,b::DistributedCellField) = ($op)(a.cellfield,b)
    ($op)(a::DistributedCellField,b::TransientDistributedCellField) = ($op)(a,b.cellfield)
    ($op)(a::TransientDistributedCellField,b::Number) = ($op)(a.cellfield,b)
    ($op)(a::Number,b::TransientDistributedCellField) = ($op)(a,b.cellfield)
    ($op)(a::TransientDistributedCellField,b::Function) = ($op)(a.cellfield,b)
    ($op)(a::Function,b::TransientDistributedCellField) = ($op)(a,b.cellfield)
  end
end

Base.broadcasted(f,a::TransientDistributedCellField,b::TransientDistributedCellField) = broadcasted(f,a.cellfield,b.cellfield)
Base.broadcasted(f,a::TransientDistributedCellField,b::DistributedCellField) = broadcasted(f,a.cellfield,b)
Base.broadcasted(f,a::DistributedCellField,b::TransientDistributedCellField) = broadcasted(f,a,b.cellfield)
Base.broadcasted(f,a::Number,b::TransientDistributedCellField) = broadcasted(f,a,b.cellfield)
Base.broadcasted(f,a::TransientDistributedCellField,b::Number) = broadcasted(f,a.cellfield,b)
Base.broadcasted(f,a::Function,b::TransientDistributedCellField) = broadcasted(f,a,b.cellfield)
Base.broadcasted(f,a::TransientDistributedCellField,b::Function) = broadcasted(f,a.cellfield,b)
Base.broadcasted(a::typeof(*),b::typeof(∇),f::TransientDistributedCellField) = broadcasted(a,b,f.cellfield)
Base.broadcasted(a::typeof(*),s::Fields.ShiftedNabla,f::TransientDistributedCellField) = broadcasted(a,s,f.cellfield)

dot(::typeof(∇),f::TransientDistributedCellField) = dot(∇,f.cellfield)
outer(::typeof(∇),f::TransientDistributedCellField) = outer(∇,f.cellfield)
outer(f::TransientDistributedCellField,::typeof(∇)) = outer(f.cellfield,∇)
cross(::typeof(∇),f::TransientDistributedCellField) = cross(∇,f.cellfield)

# Skeleton related
function Base.getproperty(f::TransientDistributedCellField, sym::Symbol)
  if sym in (:⁺,:plus,:⁻, :minus)
    derivatives = ()
    cellfield = DistributedCellField(f.cellfield,sym)
    for iderivative in f.derivatives
      derivatives = (derivatives...,DistributedCellField(iderivative))
    end
    return TransientSingleFieldCellField(cellfield,derivatives)
  else
    return getfield(f, sym)
  end
end

Base.propertynames(x::TransientDistributedCellField, private::Bool=false) = propertynames(x.cellfield, private)

for op in (:outer,:*,:dot)
  @eval begin
    ($op)(a::TransientDistributedCellField,b::SkeletonPair{<:DistributedCellField}) = ($op)(a.cellfield,b)
    ($op)(a::SkeletonPair{<:DistributedCellField},b::TransientDistributedCellField) = ($op)(a,b.cellfield)
  end
end

Arrays.evaluate!(cache,k::Operation,a::TransientDistributedCellField,b::SkeletonPair{<:DistributedCellField}) = evaluate!(cache,k,a.cellfield,b)

Arrays.evaluate!(cache,k::Operation,a::SkeletonPair{<:DistributedCellField},b::TransientDistributedCellField) = evaluate!(cache,k,a,b.cellfield)

CellData.jump(a::TransientDistributedCellField) = jump(a.cellfield)
CellData.mean(a::TransientDistributedCellField) = mean(a.cellfield)
