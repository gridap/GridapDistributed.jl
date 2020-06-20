
abstract type DistributedVector{T} end
# A distributed vector is a vector that can be restricted to a part of a communicator.
# Do not use the abstract type in function arguments since we want vectors from other packages
# to be used as distributed vectors. Use duck typing.

# a distributed vector should be indexable by a distributed index set
# The restriction of the resulting object to a part in a communicator
# should be a vector indexable by the local indices in this part.
# Note: some implementations will need to change the state of a
# to perform this operation and the result can take ownership of some
# part o the state of a. Be aware of this.
function Base.getindex(a,indices::DistributedIndexSet)
  @abstractmethod
end

Base.eltype(::Type{<:DistributedVector{T}}) where T=T
Base.eltype(a::DistributedVector)=Base.eltype(typeof(a))


# Make the state of the vector globally consistent
function exchange!(a)
  @abstractmethod
end

# Build a Distributed vector from an index set
# the resulting object is assumed to be locally indexable when restricted to a part
function DistributedVector{T}(initializer::Function,indices::DistributedIndexSet,args...) where T
  @abstractmethod
end

function DistributedVector(initializer::Function,indices::DistributedIndexSet,args...)
  @abstractmethod
end

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

# Assembly related

function Gridap.Algebra.allocate_vector(::Type{V},gids::DistributedIndexSet) where V <: AbstractVector
  ngids = num_gids(gids)
  allocate_vector(V,ngids)
end

struct SequentialIJV{A,B}
  dIJV::A
  gIJV::B
end

get_distributed_data(a::SequentialIJV) = a.dIJV


function Gridap.Algebra.allocate_coo_vectors(::Type{M},dn::DistributedData) where M <: AbstractMatrix

  part_to_n = gather(dn)
  n = sum(part_to_n)
  gIJV = allocate_coo_vectors(M,n)

  _fill_offsets!(part_to_n)

  dIJV = DistributedData(get_comm(dn), part_to_n) do part, part_to_n
    spos=part_to_n[part]+1
    if (part == length(part_to_n))
      epos=n
    else
      epos=part_to_n[part+1]
    end
    map( i -> SubVector(i,spos,epos), gIJV)
  end

  SequentialIJV(dIJV,gIJV)
end

function Gridap.Algebra.finalize_coo!(
  ::Type{M},IJV::SequentialIJV,m::DistributedIndexSet,n::DistributedIndexSet) where M <: AbstractMatrix
  I,J,V = IJV.gIJV
  finalize_coo!(M,I,J,V,num_gids(m),num_gids(n))
end
