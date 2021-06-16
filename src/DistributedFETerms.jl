function Gridap.FESpaces.collect_cell_matrix(
  u::DistributedFESpace,v::DistributedFESpace,terms)
  DistributedData(u,v,terms) do part, (u,_), (v,_), terms
    collect_cell_matrix(u,v,terms)
  end
end

function Gridap.FESpaces.collect_cell_vector(v::DistributedFESpace,terms)
  DistributedData(v,terms) do part, (v,_), terms
    collect_cell_vector(v,terms)
  end
end

function Gridap.FESpaces.collect_cell_matrix_and_vector(
  u::DistributedFESpace,v::DistributedFESpace,mterms,vterms)
  DistributedData(u,v,mterms,vterms) do part, (u,_), (v,_), mterms, vterms
    collect_cell_matrix_and_vector(u,v,mterms,vterms)
  end
end

function Gridap.FESpaces.collect_cell_matrix_and_vector(
  u::DistributedFESpace,v::DistributedFESpace,mterms,vterms,uhd::DistributedFEFunction)
  DistributedData(u,v,mterms,vterms,uhd) do part, (u,_), (v,_), mterms, vterms, uhd
    collect_cell_matrix_and_vector(u,v,mterms,vterms,uhd)
  end
end
