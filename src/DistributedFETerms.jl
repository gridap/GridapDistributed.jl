
function Gridap.FESpaces.collect_cell_jacobian(
  uh,du::DistributedCellBasis,v::DistributedCellBasis,terms)
  DistributedData(uh,du,v,terms) do part, uh, du, v, terms
    collect_cell_jacobian(uh,du,v,terms)
  end
end

function Gridap.FESpaces.collect_cell_matrix(
  u::DistributedCellBasis,v::DistributedCellBasis,terms)
  DistributedData(u,v,terms) do part, u, v, terms
    collect_cell_matrix(u,v,terms)
  end
end

function Gridap.FESpaces.collect_cell_residual(
  uh,v::DistributedCellBasis,terms)
  DistributedData(uh,v,terms) do part, uh, v, terms
    collect_cell_residual(uh,v,terms)
  end
end

function Gridap.FESpaces.collect_cell_vector(
  uhd,v::DistributedCellBasis,terms)
  DistributedData(uhd,v,terms) do part, uhd, v, terms
    collect_cell_vector(uhd,v,terms)
  end
end

function Gridap.FESpaces.collect_cell_matrix_and_vector(
  uhd,u::DistributedCellBasis,v::DistributedCellBasis,terms)
  DistributedData(uhd,u,v,terms) do part, uhd, u, v, terms
    collect_cell_matrix_and_vector(uhd,u,v,terms)
  end
end

function Gridap.FESpaces.collect_cell_jacobian_and_residual(
  uh::DistributedFEFunction,du::DistributedCellBasis,v::DistributedCellBasis,terms)
  DistributedData(uh,du,v,terms) do part, uh, du, v, terms
    collect_cell_jacobian_and_residual(uh,du,v,terms)
  end
end
