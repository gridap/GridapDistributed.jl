
abstract type GloballyAddressableVector{T} end

function GloballyAddressableVector{T}(
  initializer::Function,comm::Communicator,nparts::Integer) where T
  @abstractmethod
end

abstract type GloballyAddressableVectorPart{T} end

Base.eltype(::Type{<:GloballyAddressableVectorPart{T}}) where T = T
Base.eltype(::GloballyAddressableVectorPart{T}) where T = T

function Gridap.FESpaces.allocate_vector(
  ::Type{<:GloballyAddressableVectorPart},
  allocator)
  @abstractmethod
end

function Gridap.Algebra.fill_entries!(
  A::GloballyAddressableVectorPart,
  v::Number)
  @abstractmethod
end

function Gridap.Algebra.add_entry!(
  A::GloballyAddressableVectorPart,
  v::Number,
  global_i::Integer,
  combine::Function=+)
  @abstractmethod
end

abstract type GloballyAddressableMatrix{T} end

function GloballyAddressableMatrix{T}(
  initializer::Function,::Communicator,nparts::Integer,args...) where T
  @abstractmethod
end

abstract type GloballyAddressableMatrixPart{T} end

function Gridap.Algebra.allocate_coo_vectors(
  ::Type{<:GloballyAddressableMatrixPart},
  l::Integer)
  @abstractmethod
end

@inline function Gridap.Algebra.is_entry_stored(
  ::Type{<:GloballyAddressableMatrixPart},
  global_i::Integer,
  global_j::Integer)
  @abstractmethod
end

@inline function Gridap.Algebra.push_coo!(
   ::Type{<:GloballyAddressableMatrixPart},
   global_I::Vector,
   global_J::Vector,
   V::Vector,
   global_ik::Integer,
   global_jk::Integer,
   vk::Number)
  @abstractmethod
end

function Gridap.Algebra.finalize_coo!(
  ::Type{<:GloballyAddressableMatrixPart},
  global_I::Vector,
  global_J::Vector,
  V::Vector)
  #TODO to think the api of this one
  @abstractmethod
end

function Gridap.Algebra.sparse_from_coo(
  ::Type{<:GloballyAddressableMatrixPart},
  global_I::Vector,
  global_J::Vector,
  V::Vector)
  #TODO to think the api of this one
  @abstractmethod
end

function Gridap.Algebra.add_entry!(
  A::GloballyAddressableMatrixPart,
  v::Number,
  global_i::Integer,
  global_j::Integer,
  combine::Function=+)
  @abstractmethod
end

function Gridap.Algebra.fill_entries!(
  A::GloballyAddressableMatrixPart,
  v::Number)
  @abstractmethod
end

function Gridap.Algebra.copy_entries!(
  a::GloballyAddressableMatrixPart,
  b::GloballyAddressableMatrixPart)
  @abstractmethod
end

struct SequentialGloballyAddressableVector{T} <: GloballyAddressableVector{T}
  parts::Vector{Vector{T}}
  vec::Vector{T}
end

function GloballyAddressableVector{T}(
  initializer::Function,comm::SequentialCommunicator,nparts::Integer,args...) where T
  parts = [initializer(i,map(a->a.parts[i],args)...) for i in 1:nparts]
  vec = sum(parts)
  parts = [vec for i in 1:nparts]
  SequentialGloballyAddressableVector(parts,vec)
end

#TODO move to Gridap
function Gridap.Algebra.add_entry!(a,v,i,combine=+)
  ai = a[i]
  a[i] = combine(ai,v)
end

function Gridap.FESpaces.allocate_vector(
  ::Type{Vector{T}}, gids::GhostedVectorPart) where T
  zeros(T,gids.ngids)
end
