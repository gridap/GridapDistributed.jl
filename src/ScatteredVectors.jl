
abstract type ScatteredVector{T} <: DistributedData end

Base.eltype(::Type{<:ScatteredVector{T}}) where T = T
Base.eltype(::ScatteredVector{T}) where T = T

get_part_type(::Type{<:ScatteredVector{T}}) where T = T
get_part_type(::ScatteredVector{T}) where T = T

function ScatteredVector{T}(initializer::Function,::Communicator,nparts::Integer,args...) where T
  @abstractmethod
end

function ScatteredVector{T}(initializer::Function,args...) where T
  comm = get_comm(get_distributed_data(first(args)))
  nparts = num_parts(get_distributed_data(first(args)))
  ScatteredVector{T}(initializer,comm,nparts,args...)
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

function scatter(comm::Communicator,v,nparts::Integer)
  if i_am_master(comm)
    part_to_v = fill(v,nparts)
  else
    T = eltype(v)
    part_to_v = T[]
  end
  scatter(comm,part_to_v)
end

struct SequentialScatteredVector{T} <: ScatteredVector{T}
  parts::Vector{T}
end

get_comm(a::SequentialScatteredVector) = SequentialCommunicator()

function ScatteredVector{T}(initializer::Function,::SequentialCommunicator,nparts::Integer,args...) where T
  parts = [initializer(i,map(a->get_distributed_data(a).parts[i],args)...) for i in 1:nparts]
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
  largs = map(a->get_distributed_data(a).part,args)
  i = get_part(comm)
  part = initializer(i,largs...)
  MPIScatteredVector{T}(part,comm)
end
