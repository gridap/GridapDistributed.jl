abstract type Communicator end

function num_parts(::Communicator)
  @abstractmethod
end

function num_workers(::Communicator)
  @abstractmethod
end

function do_on_parts(task::Function,::Communicator,args...)
  @abstractmethod
end

function do_on_parts(task::Function,args...)
  comm = get_comm(get_distributed_data(first(args)))
  do_on_parts(task,comm,args...)
end

function i_am_master(::Communicator)
  @abstractmethod
end

# All communicators that are to be executed in the master to workers
# model inherit from these one
abstract type OrchestratedCommunicator <: Communicator end

function i_am_master(::OrchestratedCommunicator)
  true
end

# This is for the communicators to be executed in MPI mode
abstract type CollaborativeCommunicator <: Communicator end

# Specializations

struct SequentialCommunicator <: OrchestratedCommunicator
  nparts::Int
end

function SequentialCommunicator(nparts::Tuple)
  SequentialCommunicator(prod(nparts))
end

function num_parts(a::SequentialCommunicator)
  a.nparts
end

function num_workers(a::SequentialCommunicator)
  1
end

function do_on_parts(task::Function,a::SequentialCommunicator,args...)
  for part in 1:num_parts(a)
    largs = map(a->get_distributed_data(a).parts[part],args)
    task(part,largs...)
  end
end

struct MPICommunicator <: CollaborativeCommunicator
  comm::MPI.Comm
  master_rank::Int
  function MPICommunicator(comm::MPI.Comm,master_rank::Int=0)
    new(comm,master_rank)
  end
end

function MPICommunicator()
  # TODO copy the communicator
  MPICommunicator(MPI.COMM_WORLD)
end

function i_am_master(comm::MPICommunicator)
  MPI.Comm_rank(comm.comm) == comm.master_rank
end

function do_on_parts(task::Function,comm::MPICommunicator,args...)
  part = get_part(comm)
  largs = map(a->get_distributed_data(a).part,args)
  task(part,largs...)
end

function num_parts(comm::MPICommunicator)
  @notimplementedif comm.comm !== MPI.COMM_WORLD
  MPI.Comm_size(comm.comm)
end

function num_workers(comm::MPICommunicator)
  MPI.Comm_size(comm.comm)
end

function get_part(comm::MPICommunicator)
  @notimplementedif comm.comm !== MPI.COMM_WORLD
  MPI.Comm_rank(comm.comm) + 1
end

