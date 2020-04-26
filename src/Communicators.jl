
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
