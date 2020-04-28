
abstract type ScatteredVector{T} end

function ScatteredVector{T}(initializer::Function,::Communicator,nparts::Integer,args...) where T
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

function ScatteredVector{T}(initializer::Function,::SequentialCommunicator,nparts::Integer,args...) where T
  parts = [initializer(i,map(a->a.parts[i],args)...) for i in 1:nparts]
  SequentialScatteredVector(parts)
end

num_parts(a::SequentialScatteredVector) = length(a.parts)

function gather!(a::AbstractVector,b::SequentialScatteredVector)
  copyto!(a,b.parts)
end

function scatter(comm::SequentialCommunicator,b::AbstractVector)
  SequentialScatteredVector(b)
end

struct MPIScatteredVector{T} <: ScatteredVector{T}
  part::T
  comm::MPICommunicator
end

get_comm(a::MPIScatteredVector) = a.comm

num_parts(a::MPIScatteredVector) = num_parts(a.comm)

function ScatteredVector{T}(initializer::Function,comm::MPICommunicator,nparts::Integer,args...) where T
  @assert nparts == num_parts(comm)
  largs = map(a->a.part,args)
  i = get_part(comm)
  part = initializer(i,largs...)
  MPIScatteredVector{T}(part,comm)
end
