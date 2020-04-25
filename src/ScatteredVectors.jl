
abstract type ScatteredVector{T} end

function ScatteredVector{T}(::Communicator,length::Integer,initializer::Function) where T
  @abstractmethod
end

function get_comm(::ScatteredVector)
  @abstractmethod
end

struct SequentialScatteredVector{T} <: ScatteredVector{T}
  data::Vector{T}
end

get_comm(a::SequentialScatteredVector) = SequentialCommunicator()

function ScatteredVector{T}(::SequentialCommunicator,length::Integer,initializer::Function) where T
  data = [initializer(i) for i in 1:length]
  SequentialScatteredVector(data)
end



