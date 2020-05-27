
struct DistributedAssemblyStrategy
  strategies::DistributedData{<:AssemblyStrategy}
end

function get_distributed_data(dstrategy::DistributedAssemblyStrategy)
  dstrategy.strategies
end

struct DistributedAssembler{M,V} <: Assembler
  matrix_type::Type{M}
  vector_type::Type{V}
  trial::DistributedFESpace
  test::DistributedFESpace
  assems::DistributedData{<:Assembler}
  strategy::DistributedAssemblyStrategy
end

function get_distributed_data(dassem::DistributedAssembler)
  dassem.assems
end

function Gridap.FESpaces.get_test(a::DistributedAssembler)
  a.test
end

function Gridap.FESpaces.get_trial(a::DistributedAssembler)
  a.trial
end

function Gridap.FESpaces.get_assembly_strategy(a::DistributedAssembler)
  a.strategy
end

function Gridap.FESpaces.allocate_matrix(dassem::DistributedAssembler,dmatdata)

  dn = DistributedData(dassem,dmatdata) do part, assem, matdata
    count_matrix_nnz_coo(assem,matdata)
  end

  dIJV = allocate_coo_vectors(dassem.matrix_type,dn)

  do_on_parts(dassem,dIJV,dmatdata) do part, assem, IJV, matdata
    I,J,V = IJV
    fill_matrix_coo_symbolic!(I,J,assem,matdata)
  end

  finalize_coo!(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)
  sparse_from_coo(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)

end

function Gridap.FESpaces.allocate_vector(a::DistributedAssembler,dvecdata)
  gids = a.test.gids
  allocate_vector(a.vector_type,gids)
end

function Gridap.FESpaces.allocate_matrix_and_vector(dassem::DistributedAssembler,ddata)

  dn = DistributedData(dassem,ddata) do part, assem, data
    count_matrix_and_vector_nnz_coo(assem,data)
  end

  dIJV = allocate_coo_vectors(dassem.matrix_type,dn)
  do_on_parts(dassem,dIJV,ddata) do part, assem, IJV, data
    I,J,V = IJV
    fill_matrix_and_vector_coo_symbolic!(I,J,assem,data)
  end
  finalize_coo!(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)
  A = sparse_from_coo(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)

  gids = dassem.test.gids
  b = allocate_vector(dassem.vector_type,gids)

  A,b
end

function Gridap.FESpaces.assemble_matrix!(dmat,dassem::DistributedAssembler, dmatdata)
  fill_entries!(dmat,zero(eltype(dmat)))
  assemble_matrix_add!(dmat,dassem,dmatdata)
end

function Gridap.FESpaces.assemble_matrix_add!(dmat,dassem::DistributedAssembler, dmatdata)
  do_on_parts(dassem,dmatdata,dmat) do part, assem, matdata, mat
    assemble_matrix_add!(mat,assem,matdata)
  end
end

function Gridap.FESpaces.assemble_vector!(dvec,dassem::DistributedAssembler, dvecdata)
  fill_entries!(dvec,zero(eltype(dvec)))
  assemble_vector_add!(dvec,dassem,dvecdata)
end

function Gridap.FESpaces.assemble_vector_add!(dvec,dassem::DistributedAssembler, dvecdata)
  do_on_parts(dassem,dvecdata,dvec) do part, assem, vecdata, vec
    assemble_vector_add!(vec,assem,vecdata)
  end
end

function Gridap.FESpaces.assemble_matrix_and_vector!(dmat,dvec,dassem::DistributedAssembler, ddata)
  fill_entries!(dmat,zero(eltype(dmat)))
  fill_entries!(dvec,zero(eltype(dvec)))
  assemble_matrix_and_vector_add!(dmat,dvec,dassem,ddata)
end

function Gridap.FESpaces.assemble_matrix_and_vector_add!(dmat,dvec,dassem::DistributedAssembler, ddata)
  do_on_parts(dassem,ddata,dmat,dvec) do part, assem, data, mat, vec
    assemble_matrix_and_vector_add!(mat,vec,assem,data)
  end
end

function Gridap.FESpaces.assemble_matrix(dassem::DistributedAssembler, dmatdata)

  dn = DistributedData(dassem,dmatdata) do part, assem, matdata
    count_matrix_nnz_coo(assem,matdata)
  end

  dIJV = allocate_coo_vectors(dassem.matrix_type,dn)

  do_on_parts(dassem,dIJV,dmatdata) do part, assem, IJV, matdata
    I,J,V = IJV
    fill_matrix_coo_numeric!(I,J,V,assem,matdata)
  end

  finalize_coo!(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)
  sparse_from_coo(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)
end

function Gridap.FESpaces.assemble_vector(dassem::DistributedAssembler, dvecdata)
  vec = allocate_vector(dassem,dvecdata)
  assemble_vector!(vec,dassem,dvecdata)
  vec
end

function Gridap.FESpaces.assemble_matrix_and_vector(dassem::DistributedAssembler,ddata)

  dn = DistributedData(dassem,ddata) do part, assem, data
    count_matrix_and_vector_nnz_coo(assem,data)
  end

  gids = dassem.test.gids
  b = allocate_vector(dassem.vector_type,gids)

  dIJV = allocate_coo_vectors(dassem.matrix_type,dn)
  do_on_parts(dassem,dIJV,ddata,b) do part, assem, IJV, data, b
    I,J,V = IJV
    fill_matrix_and_vector_coo_numeric!(I,J,V,b,assem,data)
  end
  finalize_coo!(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)
  A = sparse_from_coo(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)

  A,b
end


#
# Specializations

# This is one of the usual assembly strategies in parallel FE computations
# (but not the one we have used in the parallel agfem paper)
# Each proc owns a set of matrix / vector rows (and all cols in these rows)
# Each proc computes locally all values in the owned rows
# This typically requires to loop also over ghost cells
struct RowsComputedLocally <: AssemblyStrategy
  part::Int
  gids::IndexSet
end

function Gridap.FESpaces.row_map(a::RowsComputedLocally,row)
  a.gids.lid_to_gid[row]
end

function Gridap.FESpaces.col_map(a::RowsComputedLocally,col)
  a.gids.lid_to_gid[col]
end

function Gridap.FESpaces.row_mask(a::RowsComputedLocally,row)
  a.part == a.gids.lid_to_owner[row]
end

function Gridap.FESpaces.col_mask(a::RowsComputedLocally,col)
  true
end

function RowsComputedLocally(V::DistributedFESpace)
  dgids = V.gids
  strategies = DistributedData(dgids) do part, gids
    RowsComputedLocally(part,gids)
  end
  DistributedAssemblyStrategy(strategies)
end


struct OwnedCellsStrategy <: AssemblyStrategy
  part::Int
  dof_gids::IndexSet
  cell_gids::IndexSet
end

function Gridap.FESpaces.row_map(a::OwnedCellsStrategy,row)
  a.dof_gids.lid_to_gid[row]
end

function Gridap.FESpaces.col_map(a::OwnedCellsStrategy,col)
  a.dof_gids.lid_to_gid[col]
end

function Gridap.FESpaces.row_mask(a::OwnedCellsStrategy,row)
  true
end

function Gridap.FESpaces.col_mask(a::OwnedCellsStrategy,col)
  true
end

function OwnedCellsStrategy(V::DistributedFESpace)
  ddof_gids  = V.gids
  dcell_gids = V.model.gids
  strategies = DistributedData(ddof_gids,dcell_gids) do part, dof_gids, cell_gids
    OwnedCellsStrategy(part,dof_gids,cell_gids)
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

    SparseMatrixAssembler(matrix_type,vector_type,U,V,strategy)
  end

  DistributedAssembler(matrix_type,vector_type,dtrial,dtest,assems,dstrategy)
end
