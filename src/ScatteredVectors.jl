
abstract type ScatteredVector{T} <: DistributedData end

Base.eltype(::Type{<:ScatteredVector{T}}) where T = T
Base.eltype(::ScatteredVector{T}) where T = T

get_part_type(::Type{<:ScatteredVector{T}}) where T = T
get_part_type(::ScatteredVector{T}) where T = T

function ScatteredVector{T}(initializer::Function,::Communicator,args...) where T
  @abstractmethod
end

function ScatteredVector{T}(initializer::Function,args...) where T
  comm = get_comm(get_distributed_data(first(args)))
  ScatteredVector{T}(initializer,comm,args...)
end

function gather!(a::AbstractVector,b::ScatteredVector)
  @abstractmethod
end

function gather(b::ScatteredVector{T}) where T
  if i_am_master(get_comm(b))
    a = zeros(T,num_parts(b))
  else
    a = zeros(T,0)
  end
  gather!(a,b)
  a
end

function scatter(comm::Communicator,b::AbstractVector)
  @abstractmethod
end

function scatter(comm::Communicator,v)
  if i_am_master(comm)
    part_to_v = fill(v,num_parts(comm))
  else
    T = eltype(v)
    part_to_v = T[]
  end
  scatter(comm,part_to_v)
end

struct SequentialScatteredVector{T} <: ScatteredVector{T}
  comm::SequentialCommunicator
  parts::Vector{T}
end

get_comm(a::SequentialScatteredVector) = a.comm

function ScatteredVector{T}(initializer::Function,comm::SequentialCommunicator,args...) where T
  nparts = num_parts(comm)
  parts = [initializer(i,map(a->get_distributed_data(a).parts[i],args)...) for i in 1:nparts]
  SequentialScatteredVector(comm,parts)
end

num_parts(a::SequentialScatteredVector) = length(a.parts)

function gather!(a::AbstractVector,b::SequentialScatteredVector)
  @assert length(a) == num_parts(b)
  copyto!(a,b.parts)
end

function scatter(comm::SequentialCommunicator,b::AbstractVector)
  @assert length(b) == num_parts(comm)
  SequentialScatteredVector(comm,b)
end

struct MPIScatteredVector{T} <: ScatteredVector{T}
  part::T
  comm::MPICommunicator
end

get_comm(a::MPIScatteredVector) = a.comm

num_parts(a::MPIScatteredVector) = num_parts(a.comm)

function ScatteredVector{T}(initializer::Function,comm::MPICommunicator,args...) where T
  largs = map(a->get_distributed_data(a).part,args)
  i = get_part(comm)
  part = initializer(i,largs...)
  MPIScatteredVector{T}(part,comm)
end
