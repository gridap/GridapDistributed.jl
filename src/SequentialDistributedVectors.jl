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
  indices::SequentialDistributedIndexSet) where T <: Number
  comm = get_comm(indices)
  data = DistributedData(comm,indices) do part, lindices
    Vector{T}(undef, length(lindices.lid_to_owner))
  end
  parts = data.parts
  SequentialDistributedVector(parts,indices)
end

function DistributedVector{T}(
  indices::SequentialDistributedIndexSet, length_entry :: Int ) where T <: AbstractVector{<:Number}
  comm = get_comm(indices)
  data = DistributedData(comm,indices) do part, lindices
    Vector{eltype(T)}[ Vector{eltype(T)}(undef,length_entry) for i=1:length(lindices.lid_to_owner) ]
  end
  parts = data.parts
  SequentialDistributedVector(parts,indices)
end

function DistributedVector{T}(
  indices::SequentialDistributedIndexSet, length_entries :: SequentialDistributedData ) where T <: AbstractVector{<:Number}
  comm = get_comm(indices)
  data = DistributedData(comm,indices,length_entries) do part, lindices, length_entries
    Vector{eltype(T)}[ Vector{eltype(T)}(undef,length_entries[i]) for i=1:length(lindices.lid_to_owner) ]
  end
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
  dv = DistributedVector{eltype(a)}(indices)
  do_on_parts(dv,indices,a) do part, v, indices, a
    for i=1:length(v)
      v[i]=a[indices.lid_to_gid[i]]
    end
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
