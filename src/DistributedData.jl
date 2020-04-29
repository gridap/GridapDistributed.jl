abstract type DistributedData end

function get_comm(a::DistributedData)
  @abstractmethod
end

function num_parts(a::DistributedData)
  @abstractmethod
end

# This can be defined for types that are not DistributedData
# to interpret them as DistributedData. See e.g., the DistributedDiscreteModel
function get_distributed_data(object)
  @abstractmethod
end

function get_distributed_data(a::DistributedData)
  a
end
