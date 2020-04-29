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

function get_comm(object)
  get_comm(get_distributed_data(object))
end

function num_parts(object)
  num_parts(get_distributed_data(object))
end

