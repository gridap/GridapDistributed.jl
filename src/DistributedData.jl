# Data distributed in parts of type T in a communicator
# Formerly, ScatteredVector
abstract type DistributedData{T} end

function get_comm(a::DistributedData)
  @abstractmethod
end

function num_parts(a)
  num_parts(get_comm(a))
end

# Construct a DistributedData object in a communicator
function DistributedData{T}(initializer::Function,::Communicator,args...) where T
  @abstractmethod
end

function DistributedData(initializer::Function,::Communicator,args...)
  @abstractmethod
end

# The comm argument can be omitted if it can be determined from the first
# data argument.
function DistributedData{T}(initializer::Function,args...) where T
  comm = get_comm(get_distributed_data(first(args)))
  DistributedData{T}(initializer,comm,args...)
end

function DistributedData(initializer::Function,args...)
  comm = get_comm(get_distributed_data(first(args)))
  DistributedData(initializer,comm,args...)
end

get_part_type(::Type{<:DistributedData{T}}) where T = T

get_part_type(::DistributedData{T}) where T = T

function gather!(a::AbstractVector,b::DistributedData)
  @abstractmethod
end

function gather(b::DistributedData{T}) where T
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

# return an object for which
# its restriction to the parts of a communicator is defined.
# The returned object is not necessarily an instance
# of DistributedData
# Do nothing by default.
function get_distributed_data(object)
  object
end

# Specializations

struct SequentialDistributedData{T} <: DistributedData{T}
  comm::SequentialCommunicator
  parts::Vector{T}
end

get_part(comm::SequentialCommunicator,a::SequentialDistributedData,part::Integer) = a.parts[part]

get_comm(a::SequentialDistributedData) = a.comm

function DistributedData{T}(initializer::Function,comm::SequentialCommunicator,args...) where T
  nparts = num_parts(comm)
  parts = [initializer(i,map(a->get_part(comm,get_distributed_data(a),i),args)...) for i in 1:nparts]
  SequentialDistributedData{T}(comm,parts)
end

function DistributedData(initializer::Function,comm::SequentialCommunicator,args...) where T
  nparts = num_parts(comm)
  parts = [initializer(i,map(a->get_part(comm,get_distributed_data(a),i),args)...) for i in 1:nparts]
  SequentialDistributedData(comm,parts)
end

function gather!(a::AbstractVector,b::SequentialDistributedData)
  @assert length(a) == num_parts(b)
  copyto!(a,b.parts)
end

function scatter(comm::SequentialCommunicator,b::AbstractVector)
  @assert length(b) == num_parts(comm)
  SequentialDistributedData(comm,b)
end

#struct MPIDistributedData{T} <: DistributedData{T}
#  part::T
#  comm::MPICommunicator
#end
#
#get_comm(a::MPIDistributedData) = a.comm
#
#num_parts(a::MPIDistributedData) = num_parts(a.comm)
#
#function DistributedData{T}(initializer::Function,comm::MPICommunicator,args...) where T
#  largs = map(a->get_distributed_data(a).part,args)
#  i = get_part(comm)
#  part = initializer(i,largs...)
#  MPIDistributedData{T}(part,comm)
#end
#
#function DistributedData(initializer::Function,comm::MPICommunicator,args...) where T
#  largs = map(a->get_distributed_data(a).part,args)
#  i = get_part(comm)
#  part = initializer(i,largs...)
#  MPIDistributedData(part,comm)
#end
