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

function Gridap.FESpaces.allocate_vector(a::DistributedAssembler,dterms)
  gids = a.test.gids
  allocate_vector(a.vector_type,gids)
end

function Gridap.FESpaces.assemble_vector!(dvec,dassem::DistributedAssembler, dterms)

  fill_entries!(dvec,zero(eltype(dvec)))

  do_on_parts(dassem,dterms,dvec) do part, assem, terms, vec

    U = get_trial(assem)
    V = get_test(assem)

    u0 = zero(U)
    v = get_cell_basis(V)

    vecdata = collect_cell_vector(u0,v,terms)
    assemble_vector_add!(vec,assem,vecdata...)
  end

end

function Gridap.FESpaces.assemble_vector(dassem::DistributedAssembler, dterms)
  vec = allocate_vector(dassem,dterms)
  assemble_vector!(vec,dassem,dterms)
  vec
end

function Gridap.FESpaces.allocate_matrix(dassem::DistributedAssembler,dterms)

  dmatdata = DistributedData(dassem,dterms) do part, assem, terms

    U = get_trial(assem)
    V = get_test(assem)

    u = get_cell_basis(U)
    v = get_cell_basis(V)

    collect_cell_matrix(u,v,terms)
  end

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

function Gridap.FESpaces.assemble_matrix!(dmat,dassem::DistributedAssembler, dterms)

  fill_entries!(dmat,zero(eltype(dmat)))

  do_on_parts(dassem,dterms,dmat) do part, assem, terms, mat

    U = get_trial(assem)
    V = get_test(assem)

    u = get_cell_basis(U)
    v = get_cell_basis(V)

    matdata = collect_cell_matrix(u,v,terms)
    assemble_matrix_add!(mat,assem,matdata...)
  end

end

function Gridap.FESpaces.assemble_matrix(dassem::DistributedAssembler, dterms)
  mat = allocate_matrix(dassem,dterms)
  assemble_matrix!(mat,dassem,dterms)
  mat
end


#function Gridap.FESpaces.assemble_matrix(dassem::DistributedAssembler, dterms)
#
#  comm = get_comm(dassem)
#  #TODO Float64
#  GloballyAddressableMatrix{Float64}(comm,dassem,dterms) do part, assem, terms
#
#    U = get_trial(assem)
#    V = get_test(assem)
#
#    u = get_cell_basis(U)
#    v = get_cell_basis(V)
#
#    matdata = collect_cell_matrix(u,v,terms)
#    assemble_matrix(assem,matdata...)
#  end
#
#end
#
#function Gridap.FESpaces.assemble_vector(dassem::DistributedAssembler, dterms)
#
#  comm = get_comm(dassem)
#  #TODO Float64
#  GloballyAddressableVector{Float64}(comm,dassem,dterms) do part, assem, terms
#
#    U = get_trial(assem)
#    V = get_test(assem)
#
#    u0 = zero(U)
#    v = get_cell_basis(V)
#
#    vecdata = collect_cell_vector(u0,v,terms)
#    assemble_vector(assem,vecdata...)
#  end
#
#end
#
#function Gridap.FESpaces.assemble_matrix_and_vector(
#  dassem::DistributedAssembler, dterms::DistributedFETerm...)
#  @notimplemented
#  #TODO
#end
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
function SparseMatrixAssemblerX(
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



