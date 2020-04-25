
abstract type GhostedVector{T} end

struct GhostedVectorPart{T}
  lid_to_item::Vector{T}
  lid_to_gid::Vector{Int}
  lid_to_owner::Vector{Int}
end

function get_comm(::GhostedVector)
  @abstractmethod
end

function GhostedVector{T}(
  ::Communicator,length::Integer,nparts::Integer,initializer::Function) where T
  @abstractmethod
end

struct SequentialGhostedVector{T} <: GhostedVector{T}
  data::Vector{GhostedVectorPart{T}}
end

get_comm(a::SequentialGhostedVector) = SequentialCommunicator()

function GhostedVector{T}(
  ::SequentialCommunicator,length::Integer,nparts::Integer,initializer::Function) where T

  data = [ initializer(i) for i in 1:nparts ]
  SequentialGhostedVector{T}(data)
end
