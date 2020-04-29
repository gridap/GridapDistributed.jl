
struct DistributedCellQuadrature
  quads::ScatteredVector{<:CellQuadrature}
end

function get_distributed_data(dquad::DistributedCellQuadrature)
  dquad.quads
end

function Gridap.Integration.get_coordinates(dquad::DistributedCellQuadrature)
  #TODO Any
  ScatteredVector{Any}(dquad) do part, quad
    get_coordinates(quad)
  end
end

function Gridap.Integration.get_weights(dquad::DistributedCellQuadrature)
  #TODO Any
  ScatteredVector{Any}(dquad) do part, quad
    get_weights(quad)
  end
end

function Gridap.CellQuadrature(dtrian::DistributedTriangulation,args...)
  comm = get_comm(dtrian)
  nparts = num_parts(dtrian)
  trians = ScatteredVector{Triangulation}(comm,nparts,dtrian) do part, trian
    CellQuadrature(trian,args...)
  end
  DistributedCellQuadrature(trians)
end

function Gridap.integrate(
  dintegrand::DistributedCellField,dtrian::DistributedTriangulation,dquad::DistributedCellQuadrature)
  comm = get_comm(dtrian)
  nparts = num_parts(dtrian)
  # TODO automatic detection of the type parameter
  ScatteredVector{Any}(comm,nparts,dintegrand,dtrian,dquad) do part, integrand, trian, quad
    integrate(integrand,trian,quad)
  end
end

function Gridap.integrate(
  integrand,dtrian::DistributedTriangulation,dquad::DistributedCellQuadrature)
  dintegrand = CellField(integrand,dtrian)
  integrate(dintegrand,dtrian,dquad)
end
