
abstract type ScatteredVector{T} end

function ScatteredVector{T}(::Communicator,length::Integer,initializer::Function) where T
  @abstractmethod
end

function get_comm(::ScatteredVector)
  @abstractmethod
end

function num_parts(::ScatteredVector)
  @abstractmethod
end

function gather!(a::AbstractVector,b::ScatteredVector)
  @abstractmethod
end

function gather(b::ScatteredVector{T}) where T
  a = zeros(T,num_parts(b))
  gather!(a,b)
  a
end

function scatter(comm::Communicator,b::AbstractVector)
  @abstractmethod
end

struct SequentialScatteredVector{T} <: ScatteredVector{T}
  parts::Vector{T}
end

get_comm(a::SequentialScatteredVector) = SequentialCommunicator()

function ScatteredVector{T}(::SequentialCommunicator,length::Integer,initializer::Function,args...) where T
  parts = [initializer(i,map(a->a.parts[i],args)...) for i in 1:length]
  SequentialScatteredVector(parts)
end

num_parts(a::SequentialScatteredVector) = length(a.parts)

function gather!(a::AbstractVector,b::SequentialScatteredVector)
  copyto!(a,b.parts)
end

function scatter(comm::SequentialCommunicator,b::AbstractVector)
  SequentialScatteredVector(b)
end


