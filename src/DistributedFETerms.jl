
# Not needded

struct DistributedFETerm
  terms::ScatteredVector{<:FETerm}
end

function get_distributed_data(dterm::DistributedFETerm)
  dterm.terms
end

function DistributedFETerm(initializer::Function,args...)
  terms = ScatteredVector(initializer,args...)
  DistributedFETerm(terms)
end

