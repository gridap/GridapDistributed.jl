struct DistributedAssembler
  assems::ScatteredVector{<:Assembler}
end

function get_distributed_data(dassem::DistributedAssembler)
  dassem.assems
end

struct DistributedAssemblyStrategy
  strategies::ScatteredVector{<:AssemblyStrategy}
end

function get_distributed_data(dstrategy::DistributedAssemblyStrategy)
  dstrategy.strategies
end

function Gridap.FESpaces.assemble_matrix(dassem::DistributedAssembler, dterms)

  comm = get_comm(dassem)
  #TODO Float64
  GloballyAddressableMatrix{Float64}(comm,dassem,dterms) do part, assem, terms

    U = get_trial(assem)
    V = get_test(assem)

    u = get_cell_basis(U)
    v = get_cell_basis(V)

    matdata = collect_cell_matrix(u,v,terms)
    assemble_matrix(assem,matdata...)
  end

end

function Gridap.FESpaces.assemble_vector(dassem::DistributedAssembler, dterms)

  comm = get_comm(dassem)
  #TODO Float64
  GloballyAddressableVector{Float64}(comm,dassem,dterms) do part, assem, terms

    U = get_trial(assem)
    V = get_test(assem)

    u0 = zero(U)
    v = get_cell_basis(V)

    vecdata = collect_cell_vector(u0,v,terms)
    assemble_vector(assem,vecdata...)
  end

end

function Gridap.FESpaces.assemble_matrix_and_vector(
  dassem::DistributedAssembler, dterms::DistributedFETerm...)
  @notimplemented
  #TODO
end

# Specializations

function RowsComputedLocally(V::DistributedFESpace)
  dgids = V.gids
  strategies = ScatteredVector(dgids) do part, gids
    RowsComputedLocally(part,gids.lid_to_gid,gids.lid_to_owner)
  end
  DistributedAssemblyStrategy(strategies)
end

function SparseMatrixAssemblerX(
  matrix_type::Type,
  vector_type::Type,
  dtrial::DistributedFESpace,
  dtest::DistributedFESpace,
  dstrategy::DistributedAssemblyStrategy,
  dtrial_alloc=dtrial,
  dtest_alloc=dtest)

  assems = ScatteredVector(
    dtrial.spaces,dtest.spaces,dstrategy,dtrial_alloc,dtest_alloc) do part, U, V, strategy, U_alloc, V_alloc

    SparseMatrixAssemblerX(matrix_type,vector_type,U,V,strategy,U_alloc,V_alloc)
  end

  DistributedAssembler(assems)
end



