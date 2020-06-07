struct MPIPETScDistributedData{T} <: DistributedData{T}
 part::T
 comm::MPIPETScCommunicator
end

get_comm(a::MPIPETScDistributedData) = a.comm

num_parts(a::MPIPETScDistributedData) = num_parts(a.comm)

function DistributedData{T}(initializer::Function,comm::MPIPETScCommunicator,args...) where T
 largs = map(a->get_distributed_data(a).part,args)
 i = MPI.Comm_rank(comm.comm)+1
 part = initializer(i,largs...)
 MPIPETScDistributedData{T}(part,comm)
end

function DistributedData(initializer::Function,comm::MPIPETScCommunicator,args...)
 largs = map(a->get_distributed_data(a).part,args)
 i = MPI.Comm_rank(comm.comm)+1
 part = initializer(i,largs...)
 MPIPETScDistributedData(part,comm)
end
