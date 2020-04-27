# @santiagobadia : This model seems a master-slave model more in the flavour of
# Julia native parallelism, which is not what I would expect from MPI.
# The abstract Communicator should probably be more abstract.
abstract type Communicator end

function do_on_parts(::Communicator,task::Function,args...)
  @abstractmethod
end

function i_am_master(::Communicator)
  @abstractmethod
end

function do_on_parts(task::Function,args...)
  comm = get_comm(first(args))
  do_on_parts(task,comm,args...)
end

struct SequentialCommunicator <: Communicator end

function do_on_parts(task::Function,::SequentialCommunicator,args...)
  for part in 1:length(first(args).parts)
    largs = map(a->a.parts[part],args)
    task(part,largs...)
  end
end

function i_am_master(::SequentialCommunicator)
  true
end

struct MPICommunicator <: Communicator
  comm::MPI.Comm
  master_rank::Int
  function MPICommunicator(comm::MPI.Comm,master_rank::Int=0)
    new(comm,master_rank)
  end
end

function MPICommunicator()
  MPICommunicator(MPI.COMM_WORLD)
end

function i_am_master(comm::MPICommunicator)
  MPI.Comm_rank(comm.comm) == comm.master_rank
end

function do_on_parts(task::Function,comm::MPICommunicator,args...)
  part = get_part(comm)
  largs = map(a->a.part,args)
  task(part,largs...)
end

function num_parts(comm::MPICommunicator)
  @notimplementedif comm.comm !== MPI.COMM_WORLD
  MPI.Comm_size(comm.comm)
end

function get_part(comm::MPICommunicator)
  @notimplementedif comm.comm !== MPI.COMM_WORLD
  MPI.Comm_rank(comm.comm) + 1
end


