# Specializations
struct SequentialDistributedVector{T,V<:AbstractVector{T},A,B,C}
  parts::Vector{V}
  indices::SequentialDistributedIndexSet{A,B,C}
end

function SequentialDistributedVector{T}(
  parts::Vector{<:AbstractVector{T}},indices::SequentialDistributedIndexSet) where T
  SequentialDistributedVector(parts,indices)
end

get_comm(a::SequentialDistributedVector) = get_comm(a.indices)

get_part(
  comm::SequentialCommunicator,
  a::SequentialDistributedVector,
  part::Integer) = a.parts[part]

function DistributedVector{T}(
  initializer::Function, indices::SequentialDistributedIndexSet,args...) where T
  comm = get_comm(indices)
  data = DistributedData(initializer,comm,args...)
  parts = data.parts
  SequentialDistributedVector{T}(parts,indices)
end

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

# Julia vectors
function Base.getindex(a::Vector,indices::SequentialDistributedIndexSet)
  DistributedVector(indices,indices,a) do part, indices, a
    a[indices.lid_to_gid]
  end
end

#function exchange!(a::Vector)
#  a
#end

# By default in the SequentialCommunicator arrays are globally indexable when restricted to a part
function get_part(comm::SequentialCommunicator,a::AbstractArray,part::Integer)
  a
end
