# Specializations
struct SequentialCommunicator <: OrchestratedCommunicator
  nparts::Int
end

function SequentialCommunicator(user_driver_function,nparts)
  comm=SequentialCommunicator(nparts)
  user_driver_function(comm)
end

function SequentialCommunicator(nparts::Tuple)
  SequentialCommunicator(prod(nparts))
end

# All objects to be used with this communicator need to implement this
# function
function get_part(comm::SequentialCommunicator,object,part::Integer)
  @abstractmethod
end

function get_part(comm::SequentialCommunicator,object::Number,part::Integer)
  object
end

function num_parts(a::SequentialCommunicator)
  a.nparts
end

function num_workers(a::SequentialCommunicator)
  1
end

function Base.:(==)(a::SequentialCommunicator,b::SequentialCommunicator)
  a.nparts == b.nparts
end

function do_on_parts(task::Function,comm::SequentialCommunicator,args...)
  for part in 1:num_parts(comm)
    largs = map(a->get_part(comm,get_distributed_data(a),part),args)
    task(part,largs...)
  end
end
