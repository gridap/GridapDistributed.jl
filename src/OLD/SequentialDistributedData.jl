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

function DistributedData(initializer::Function,comm::SequentialCommunicator,args...)
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
