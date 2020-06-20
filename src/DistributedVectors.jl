
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
