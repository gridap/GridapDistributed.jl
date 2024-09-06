
# Extensions to Gridap/Arrays/Tables.jl

function generate_ptrs(vv::AbstractArray{<:AbstractArray{T}}) where T
  ptrs = Vector{Int32}(undef,length(vv)+1)
  Arrays._generate_data_and_ptrs_fill_ptrs!(ptrs,vv)
  Arrays.length_to_ptrs!(ptrs)
  return ptrs
end

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
