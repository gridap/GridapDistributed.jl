
"""
    allocate_vector(::Type{V},indices) where V

Allocate a vector of type `V` indexable at the indices `indices`
"""
function Gridap.Algebra.allocate_vector(::Type{<:MPIPETScDistributedVector{Float64}},indices::MPIPETScDistributedIndexSet)
  DistributedVector{Float64}(indices) do part
     Vector{Float64}(undef,length(indices.parts.part.lid_to_gid))
  end
end

function Gridap.Algebra.allocate_vector(
  ::Type{PETSc.Vec{Float64}},
  indices::MPIPETScDistributedIndexSet,
)
  ng = num_gids(indices)
  nl = num_owned_entries(indices)
  vec=PETSc.Vec(Float64, ng; mlocal = nl, comm = get_comm(indices).comm)
  vec.insertmode = PETSc.C.ADD_VALUES
  vec
end


"""
    fill_entries!(a,v)

Fill the entries of array `a` with the value `v`. Returns `a`.
"""
function Gridap.Algebra.fill_entries!(a::MPIPETScDistributedVector{Float64},v)
  do_on_parts(get_comm(a)) do part
    fill!(a.part,v)
  end
  a
end
