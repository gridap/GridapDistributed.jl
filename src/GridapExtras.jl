
# New type of Vector allocation

"""
    TrackedArrayAllocation{T} <: GridapType

Array that keeps track of which entries have been touched. For assembly purposes.
"""
struct TrackedArrayAllocation{T}
  values  :: Vector{T}
  touched :: Vector{Bool}
end

function TrackedArrayAllocation(values)
  touched = fill(false,length(values))
  TrackedArrayAllocation(values,touched)
end

Algebra.LoopStyle(::Type{<:TrackedArrayAllocation}) = Algebra.Loop()
Algebra.create_from_nz(a::TrackedArrayAllocation) = a.values

@inline function Arrays.add_entry!(combine::Function,a::TrackedArrayAllocation,v::Nothing,i)
  if i > 0
    a.touched[i] = true
  end
  nothing
end
@inline function Arrays.add_entry!(combine::Function,a::TrackedArrayAllocation,v,i)
  if i > 0
    a.touched[i] = true
    a.values[i] = combine(v,a.values[i])
  end
  nothing
end
@inline function Arrays.add_entries!(combine::Function,a::TrackedArrayAllocation,v::Nothing,i)
  for ie in i
    Arrays.add_entry!(combine,a,nothing,ie)
  end
  nothing
end
@inline function Arrays.add_entries!(combine::Function,a::TrackedArrayAllocation,v,i)
  for (ve,ie) in zip(v,i)
    Arrays.add_entry!(combine,a,ve,ie)
  end
  nothing
end

# change_axes

function change_axes(a::Algebra.CounterCOO{T,A}, axes::A) where {T,A}
  b = Algebra.CounterCOO{T}(axes)
  b.nnz = a.nnz
  b
end

function change_axes(a::Algebra.AllocationCOO{T,A}, axes::A) where {T,A}
  counter = change_axes(a.counter,axes)
  Algebra.AllocationCOO(counter,a.I,a.J,a.V)
end
