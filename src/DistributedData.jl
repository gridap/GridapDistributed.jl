abstract type DistributedData end

function get_comm(a::DistributedData)
  @abstractmethod
end

function num_parts(a::DistributedData)
  @abstractmethod
end

function get_distributed_data(a::DistributedData)
  a
end

# Providing of some "distributed behavior" to objects that are not instances
# of `DistributedData`

# This can be defined for types that are not DistributedData
# to interpret them as DistributedData. See e.g., the DistributedDiscreteModel
function get_distributed_data(object)
  @abstractmethod
end

function get_distributed_data(comm::Communicator,object)
  get_distributed_data(object)
end

function get_comm(object)
  get_comm(get_distributed_data(object))
end

function num_parts(object)
  num_parts(get_distributed_data(object))
end

function get_distributed_data(comm::Communicator,d::Dict)

  if length(d) == 0
    return ScatteredVector(comm) do part
      Dict()
    end
  end

  thekeys = keys(d)
  thevals = values(d)

  v = first(thevals)
  comm = get_comm(get_distributed_data(v))
  nparts = num_parts(get_distributed_data(v))

  function initializer(part, vals...)
    ld = Dict()
    for (i,k) in enumerate(thekeys)
      ld[k] = vals[i]
    end
    ld
  end

  ScatteredVector(initializer,comm,nparts,values(d)...)
end

function get_distributed_data(comm::Communicator,d::Dict)
end

