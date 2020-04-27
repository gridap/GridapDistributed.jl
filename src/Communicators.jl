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
  do_on_parts(comm,task,args...)
end

struct SequentialCommunicator <: Communicator end

function do_on_parts(::SequentialCommunicator,task::Function,args...)
  for part in 1:length(first(args).parts)
    largs = map(a->a.parts[part],args)
    task(part,largs...)
  end
end

function i_am_master(::SequentialCommunicator)
  true
end
