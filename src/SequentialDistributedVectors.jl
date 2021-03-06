# Specializations
struct SequentialDistributedVector{T,V<:AbstractVector{T},A,B,C}
  parts::Vector{V}
  indices::SequentialDistributedIndexSet{A,B,C}
end

get_comm(a::SequentialDistributedVector) = get_comm(a.indices)

get_part(
  comm::SequentialCommunicator,
  a::SequentialDistributedVector,
  part::Integer) = a.parts[part]


function DistributedVector(
  initializer::Function, indices::SequentialDistributedIndexSet,args...)
  comm = get_comm(indices)
  data = DistributedData(initializer,comm,args...)
  parts = data.parts
  SequentialDistributedVector(parts,indices)
end

function Base.getindex(a::SequentialDistributedVector,indices::SequentialDistributedIndexSet)
  @notimplementedif a.indices !== indices
  exchange!(a)
  a
end

function exchange!(a::SequentialDistributedVector)
  indices = a.indices
  for part in 1:num_parts(indices)
    lid_to_gid = indices.parts.parts[part].lid_to_gid
    lid_to_item = a.parts[part]
    lid_to_owner = indices.parts.parts[part].lid_to_owner
    for lid in 1:length(lid_to_item)
      gid = lid_to_gid[lid]
      owner = lid_to_owner[lid]
      if owner != part
        lid_owner = indices.parts.parts[owner].gid_to_lid[gid]
        item = a.parts[owner][lid_owner]
        lid_to_item[lid] = item
      end
    end
  end
  a
end

# Note: this type dispatch is needed because setindex! is
#       not available for Table
function exchange!(a::SequentialDistributedVector{T,<:Table}) where {T}
  indices = a.indices
  for part in 1:num_parts(indices)
    lid_to_gid = indices.parts.parts[part].lid_to_gid
    lid_to_item = a.parts[part]
    lid_to_owner = indices.parts.parts[part].lid_to_owner
    for lid in 1:length(lid_to_item)
      gid = lid_to_gid[lid]
      owner = lid_to_owner[lid]
      if owner != part
        lid_owner = indices.parts.parts[owner].gid_to_lid[gid]
        k=a.parts[owner].ptrs[lid_owner]
        for j=lid_to_item.ptrs[lid]:lid_to_item.ptrs[lid+1]-1
          lid_to_item.data[j]=a.parts[owner].data[k]
          k=k+1
        end
      end
    end
  end
  a
end


# Julia vectors
function Base.getindex(a::Vector,indices::SequentialDistributedIndexSet)
  dv = DistributedVector(indices,indices,a) do part, indices, a
    v=Vector{eltype(a)}(undef,length(indices.lid_to_gid))
    for i=1:length(v)
      v[i]=a[indices.lid_to_gid[i]]
    end
    v
  end
  dv
end

#function exchange!(a::Vector)
#  a
#end

# By default in the SequentialCommunicator arrays are globally indexable when restricted to a part
function get_part(comm::SequentialCommunicator,a::AbstractArray,part::Integer)
  a
end
