
abstract type GloballyAddressableVector{T} <: DistributedData end

function GloballyAddressableVector{T}(
  initializer::Function,comm::Communicator) where T
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

abstract type GloballyAddressableMatrix{T} <: DistributedData end

function GloballyAddressableMatrix{T}(
  initializer::Function,::Communicator,args...) where T
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
  V::Vector,
  row_alloc,
  col_alloc)
  @abstractmethod
end

function Gridap.Algebra.sparse_from_coo(
  ::Type{<:GloballyAddressableMatrixPart},
  global_I::Vector,
  global_J::Vector,
  V::Vector,
  row_alloc,
  col_alloc)
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
  comm::SequentialCommunicator
  parts::Vector{Vector{T}}
  vec::Vector{T}
end

get_comm(a::SequentialGloballyAddressableVector) = a.comm

num_parts(a::SequentialGloballyAddressableVector) = length(a.parts)

function GloballyAddressableVector{T}(
  initializer::Function,comm::SequentialCommunicator,args...) where T
  nparts = num_parts(comm)
  parts = [initializer(i,map(a->get_distributed_data(comm,a).parts[i],args)...) for i in 1:nparts]
  vec = sum(parts)
  parts = [vec for i in 1:nparts]
  SequentialGloballyAddressableVector(comm,parts,vec)
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

struct SequentialGloballyAddressableMatrix{T,M<:AbstractMatrix{T}} <: GloballyAddressableMatrix{T}
  comm::SequentialCommunicator
  parts::Vector{M}
  mat::M
end

get_comm(a::SequentialGloballyAddressableMatrix) = a.comm

num_parts(a::SequentialGloballyAddressableMatrix) = length(a.parts)

function GloballyAddressableMatrix{T}(
  initializer::Function,comm::SequentialCommunicator,args...) where T
  nparts = num_parts(comm)
  parts = [initializer(i,map(a->get_distributed_data(comm,a).parts[i],args)...) for i in 1:nparts]
  mat = sum(parts)
  parts = [mat for i in 1:nparts]
  SequentialGloballyAddressableMatrix(comm,parts,mat)
end

# TODO the following have to be implemented for AbstractMatrix
# instead of SparseMatrixCSC when enhanced the interface in Gridap
using SparseArrays

function Gridap.Algebra.finalize_coo!(
  ::Type{M},
  global_I::Vector,
  global_J::Vector,
  V::Vector,
  row_gids::GhostedVectorPart,
  col_gids::GhostedVectorPart) where M <:SparseMatrixCSC

  n = row_gids.ngids
  m = col_gids.ngids
  finalize_coo!(M,global_I,global_J,V,n,m)
end

function Gridap.Algebra.sparse_from_coo(
  ::Type{<:M},
  global_I::Vector,
  global_J::Vector,
  V::Vector,
  row_gids::GhostedVectorPart,
  col_gids::GhostedVectorPart) where M <:SparseMatrixCSC

  n = row_gids.ngids
  m = col_gids.ngids
  sparse_from_coo(M,global_I,global_J,V,n,m)
end


