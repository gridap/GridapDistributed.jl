
struct DistributedCellQuadrature
  quads::ScatteredVector{<:CellQuadrature}
end

function get_distributed_data(dquad::DistributedCellQuadrature)
  dquad.quads
end

function Gridap.CellQuadrature(dtrian::DistributedTriangulation,args...)
  comm = get_comm(dtrian)
  nparts = num_parts(dtrian)
  trians = ScatteredVector{Triangulation}(comm,nparts,dtrian) do part, trian
    CellQuadrature(trian,args...)
  end
  DistributedCellQuadrature(trians)
end

function Gridap.integrate(dintegrand,dtrian::DistributedTriangulation,dquad::DistributedCellQuadrature)
  comm = get_comm(dtrian)
  nparts = num_parts(dtrian)
  # TODO automatic detection of the type parameter
  ScatteredVector{Any}(comm,nparts,dintegrand,dtrian,dquad) do part, integrand, trian, quad
    integrate(integrand,trian,quad)
  end
end
