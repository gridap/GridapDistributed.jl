struct MPIPETScDistributedData{T} <: DistributedData{T}
 part::T
 comm::MPIPETScCommunicator
end

get_comm(a::MPIPETScDistributedData) = a.comm

num_parts(a::MPIPETScDistributedData) = num_parts(a.comm)

get_part(comm::MPIPETScCommunicator,a::MPIPETScDistributedData,part::Integer) = a.part

function DistributedData{T}(initializer::Function,comm::MPIPETScCommunicator,args...) where T
 i = MPI.Comm_rank(comm.comm)+1
 largs = map(a->get_part(comm,get_distributed_data(a),i),args)
 part = initializer(i,largs...)
 MPIPETScDistributedData{T}(part,comm)
end

function DistributedData(initializer::Function,comm::MPIPETScCommunicator,args...)
 i = MPI.Comm_rank(comm.comm)+1
 largs = map(a->get_part(comm,get_distributed_data(a),i),args)
 part = initializer(i,largs...)
 MPIPETScDistributedData(part,comm)
end

function gather!(a::AbstractVector,b::MPIPETScDistributedData)
  comm=get_comm(b)
  if (i_am_master(comm))
    @assert length(a) == num_parts(b)
    a[comm.master_rank+1]=b.part
    MPI.Gather!(nothing    ,       a, 1, comm.master_rank, comm.comm)
  else
    MPI.Gather!(Ref(b.part), nothing, 1, comm.master_rank, comm.comm)
  end
end

function scatter(comm::MPIPETScCommunicator,b::AbstractVector)
  if (i_am_master(comm)) @assert length(b) == num_parts(comm) end
  v=MPI.Scatter(b,1,comm.master_rank,comm.comm)
  DistributedData(comm) do part
    v[1]
  end
end
