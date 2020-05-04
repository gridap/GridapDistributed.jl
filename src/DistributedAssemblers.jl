struct DistributedAssemblyStrategy
  strategies::DistributedData{<:AssemblyStrategy}
end

function get_distributed_data(dstrategy::DistributedAssemblyStrategy)
  dstrategy.strategies
end

struct DistributedAssembler{M,V}
  matrix_type::Type{M}
  vector_type::Type{V}
  trial::DistributedFESpace
  test::DistributedFESpace
  assems::DistributedData{<:Assembler}
end

function get_distributed_data(dassem::DistributedAssembler)
  dassem.assems
end

function Gridap.FESpaces.allocate_vector(a::DistributedAssembler,dvecdata)
  gids = a.test.gids
  allocate_vector(a.vector_type,gids)
end

function Gridap.FESpaces.assemble_vector!(dvec,dassem::DistributedAssembler, dvecdata)
  fill_entries!(dvec,zero(eltype(dvec)))
  do_on_parts(dassem,dvecdata,dvec) do part, assem, vecdata, vec
    assemble_vector_add!(vec,assem,vecdata...)
  end
end

function Gridap.FESpaces.assemble_vector(dassem::DistributedAssembler, dvecdata)
  vec = allocate_vector(dassem,dvecdata)
  assemble_vector!(vec,dassem,dvecdata)
  vec
end

function Gridap.FESpaces.allocate_matrix(dassem::DistributedAssembler,dmatdata)

  dn = DistributedData(dassem,dmatdata) do part, assem, matdata
    count_matrix_nnz_coo(assem,matdata...)
  end

  dIJV = allocate_coo_vectors(dassem.matrix_type,dn)

  do_on_parts(dassem,dIJV,dmatdata) do part, assem, IJV, matdata
    I,J,V = IJV
    fill_matrix_coo_symbolic!(I,J,assem,matdata...)
  end

  finalize_coo!(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)
  sparse_from_coo(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)

end

function Gridap.FESpaces.assemble_matrix!(dmat,dassem::DistributedAssembler, dmatdata)
  fill_entries!(dmat,zero(eltype(dmat)))
  do_on_parts(dassem,dmatdata,dmat) do part, assem, matdata, mat
    assemble_matrix_add!(mat,assem,matdata...)
  end
end

function Gridap.FESpaces.assemble_matrix(dassem::DistributedAssembler, dmatdata)
  mat = allocate_matrix(dassem,dmatdata)
  assemble_matrix!(mat,dassem,dmatdata)
  mat
end


#
# Specializations


function RowsComputedLocally(V::DistributedFESpace)
  dgids = V.gids
  strategies = DistributedData(dgids) do part, gids
    RowsComputedLocally(part,gids.lid_to_gid,gids.lid_to_owner)
  end
  DistributedAssemblyStrategy(strategies)
end

# TODO this assumes that the global matrix type is the same
# as the local one
function Gridap.FESpaces.SparseMatrixAssembler(
  matrix_type::Type,
  vector_type::Type,
  dtrial::DistributedFESpace,
  dtest::DistributedFESpace,
  dstrategy::DistributedAssemblyStrategy)

  assems = DistributedData(
    dtrial.spaces,dtest.spaces,dstrategy) do part, U, V, strategy

    SparseMatrixAssemblerX(matrix_type,vector_type,U,V,strategy)
  end

  DistributedAssembler(matrix_type,vector_type,dtrial,dtest,assems)
end



