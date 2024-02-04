
function Fields.gradient(F::FESpaces.IntegrandWithMeasure,uh::Vector{<:DistributedCellField},K::Int)
  @check 0 < K <= length(uh)
  local_fields   = map(local_views,uh) |> to_parray_of_arrays
  local_measures = map(local_views,F.dΩ) |> to_parray_of_arrays
  contribs = map(local_measures,local_fields) do dΩ,lf
    _f(uk) = F.F(lf[1:K-1]...,uk,lf[K+1:end]...,dΩ...)
    return Fields.gradient(_f,lf[K])
  end
  return DistributedDomainContribution(contribs)
end

function Fields.jacobian(F::FESpaces.IntegrandWithMeasure,uh::Vector{<:DistributedCellField},K::Int)
  @check 0 < K <= length(uh)
  local_fields   = map(local_views,uh) |> to_parray_of_arrays
  local_measures = map(local_views,F.dΩ) |> to_parray_of_arrays
  contribs = map(local_measures,local_fields) do dΩ,lf
    _f(uk) = F.F(lf[1:K-1]...,uk,lf[K+1:end]...,dΩ...)
    return Fields.jacobian(_f,lf[K])
  end
  return DistributedDomainContribution(contribs)
end
