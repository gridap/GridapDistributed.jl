module GloballyAddressableArraysTests

using GridapDistributed


T = Float64
nparts = 4
n = 10

comm = SequentialCommunicator(nparts)

v = GloballyAddressableVector{T}(comm,nparts) do part
  rand(T,n)
end

#display(v.parts)

using Gridap.Algebra

w = v.vec
fill_entries!(w,zero(eltype(w)))

add_entry!(w,4,2)
add_entry!(w,3,2)

#display(v.parts)




end # module
