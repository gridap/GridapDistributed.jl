
struct DistributedAssemblyStrategy{T<:AssemblyStrategy}
  strategies::DistributedData{T}
end

function get_distributed_data(dstrategy::DistributedAssemblyStrategy)
  dstrategy.strategies
end

struct DistributedAssembler{M,V,AS} <: Assembler
  matrix_type             :: Type{M}
  vector_type             :: Type{V}
  assembly_strategy_type  :: Type{AS}
  trial::DistributedFESpace
  test::DistributedFESpace
  assems::DistributedData{<:Assembler}
  strategy::DistributedAssemblyStrategy{AS}
end

"""
    allocate_local_vector(::Type{AS}, ::Type{V}, indices) where {AS,V}

Allocate the local vector required in order to assemble a global vector
of type V accordingly to the assembly algorithm underlying the assembly strategy
AS.  The global vector is indexable using indices.
"""
function allocate_local_vector(::Type{AS}, ::Type{V}, indices::DistributedIndexSet) where {AS,V}
   @abstractmethod
end

"""
    get_local_matrix_type(::Type{M}) where M

Given a global matrix type M, returns the local matrix type
whose assembler-related methods are appropiate for the global
assembly process of M.

"""
function get_local_matrix_type(::Type{M}) where M
  @abstractmethod
end

function get_local_matrix_type(::Type{M}) where M <: Union{<:SparseMatrixCSR,<:SparseMatrixCSC}
  M
end

"""
    get_local_vector_type(::Type{V}) where V

Given a global vector type V, returns the local vector type
whose assembler-related methods are appropiate for the global
assembly process of V.

"""
function get_local_vector_type(::Type{V}) where V
  @abstractmethod
end

function get_local_vector_type(::Type{V}) where V <: Vector
  V
end

function assemble_global_matrix(::Type{AS}, ::Type{M},
                                ::DistributedData,
                                ::DistributedIndexSet,
                                ::DistributedIndexSet) where {AS,M}
   @abstractmethod
end

function assemble_global_vector(::Type{AS}, ::Type{V},
                                ::DistributedData,
                                ::DistributedIndexSet) where {AS,V}
   @abstractmethod
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
  A = assemble_global_matrix(dassem.assembly_strategy_type,
                             dassem.matrix_type,
                             dIJV,
                             dassem.test.gids,
                             dassem.trial.gids)
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
  A = assemble_global_matrix(dassem.assembly_strategy_type,
                             dassem.matrix_type,
                             dIJV,
                             dassem.test.gids,
                             dassem.trial.gids)
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
  A = assemble_global_matrix(dassem.assembly_strategy_type,
                             dassem.matrix_type,
                             dIJV,
                             dassem.test.gids,
                             dassem.trial.gids)
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
  db = allocate_local_vector(dassem.assembly_strategy_type,dassem.vector_type,gids)

  dIJV = allocate_coo_vectors(dassem.matrix_type,dn)
  do_on_parts(dassem,dIJV,ddata,db) do part, assem, IJV, data, b
    I,J,V = IJV
    fill_matrix_and_vector_coo_numeric!(I,J,V,b,assem,data)
  end
  finalize_coo!(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)

  A = assemble_global_matrix(dassem.assembly_strategy_type,
                             dassem.matrix_type,
                             dIJV,
                             dassem.test.gids,
                             dassem.trial.gids)

  b = assemble_global_vector(dassem.assembly_strategy_type,
                             dassem.vector_type,
                             db,
                             dassem.test.gids)
  A,b
end


#
# Specializations

# This is one of the usual assembly strategies in parallel FE computations
# (but not the one we have used in the parallel agfem paper)
# Each proc owns a set of matrix / vector rows (and all cols in these rows)
# Each proc computes locally all values in the owned rows
# This typically requires to loop also over ghost cells
struct RowsComputedLocally{GlobalDoFs} <: AssemblyStrategy
  part::Int
  gids::IndexSet
end

function Gridap.FESpaces.row_map(a::RowsComputedLocally{true},row)
  a.gids.lid_to_gid[row]
end

function Gridap.FESpaces.col_map(a::RowsComputedLocally{true},col)
  a.gids.lid_to_gid[col]
end

function Gridap.FESpaces.row_map(a::RowsComputedLocally{false},row)
  row
end

function Gridap.FESpaces.col_map(a::RowsComputedLocally{false},col)
  col
end

function Gridap.FESpaces.row_mask(a::RowsComputedLocally,row)
  a.part == a.gids.lid_to_owner[row]
end

function Gridap.FESpaces.col_mask(a::RowsComputedLocally,col)
  true
end

function RowsComputedLocally(V::DistributedFESpace; global_dofs=true)
   dgids = V.gids
   strategies = DistributedData(dgids) do part, gids
     RowsComputedLocally{global_dofs}(part,gids)
   end
   DistributedAssemblyStrategy(strategies)
end

struct OwnedCellsStrategy{GlobalDoFs} <: AssemblyStrategy
  part::Int
  dof_gids::IndexSet
  cell_gids::IndexSet
end

function Gridap.FESpaces.row_map(a::OwnedCellsStrategy{true},row)
  a.dof_gids.lid_to_gid[row]
end

function Gridap.FESpaces.col_map(a::OwnedCellsStrategy{true},col)
  a.dof_gids.lid_to_gid[col]
end

function Gridap.FESpaces.row_map(a::OwnedCellsStrategy{false},row)
  row
end

function Gridap.FESpaces.col_map(a::OwnedCellsStrategy{false},col)
  col
end

function Gridap.FESpaces.row_mask(a::OwnedCellsStrategy,row)
  true
end

function Gridap.FESpaces.col_mask(a::OwnedCellsStrategy,col)
  true
end

function OwnedCellsStrategy(M::DistributedDiscreteModel, V::DistributedFESpace; global_dofs=true)
  dcell_gids = M.gids
  ddof_gids  = V.gids
  strategies = DistributedData(ddof_gids,dcell_gids) do part, dof_gids, cell_gids
    OwnedCellsStrategy{global_dofs}(part,dof_gids,cell_gids)
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
  dstrategy::DistributedAssemblyStrategy{T}) where T

  assems = DistributedData(
    dtrial.spaces,dtest.spaces,dstrategy) do part, U, V, strategy
    SparseMatrixAssembler(get_local_matrix_type(matrix_type),get_local_vector_type(vector_type),U,V,strategy)
  end

  DistributedAssembler(matrix_type,
                       vector_type,
                       T,
                       dtrial,
                       dtest,
                       assems,
                       dstrategy)
end

function allocate_local_vector(
  ::Union{Type{RowsComputedLocally{false}},Type{OwnedCellsStrategy{false}}},
  ::Type{V},
  indices::SequentialDistributedIndexSet,
) where V<:Vector
  DistributedData(indices) do part,index
   T = get_local_vector_type(V)
   lvec=T(undef,length(index.lid_to_gid))
   fill!(lvec,zero(eltype(T)))
   lvec
  end
end

function allocate_local_vector(
  ::Union{Type{RowsComputedLocally{true}},Type{OwnedCellsStrategy{true}}},
  ::Type{V},
  indices::SequentialDistributedIndexSet,
) where V<:Vector
   T = get_local_vector_type(V)
   vec=T(undef,num_gids(indices))
   fill!(vec,zero(eltype(T)))
   vec
end


function assemble_global_matrix(::Union{Type{RowsComputedLocally{T}},Type{OwnedCellsStrategy{T}}},
                                ::Type{M},
                                IJV::SequentialIJV,
                                m::DistributedIndexSet,
                                n::DistributedIndexSet) where {T,M}
  if (!T)
     do_on_parts(IJV.dIJV,m,n) do part, IJV, mindexset, nindexset
        I,J,V = IJV
        for i=1:length(I)
          I[i]=mindexset.lid_to_gid[I[i]]
          J[i]=nindexset.lid_to_gid[J[i]]
        end
     end
  end
  I,J,V = IJV.gIJV
  A=sparse_from_coo(M,I,J,V,num_gids(m),num_gids(n))
end

function assemble_global_vector(::Union{Type{RowsComputedLocally{false}},Type{OwnedCellsStrategy{false}}},
                                ::Type{M},
                                db::DistributedData,
                                m::DistributedIndexSet) where M <: Vector
  b=allocate_vector(M, num_gids(m))
  do_on_parts(m, db, b) do part, mindexset, blocal, b
    for i=1:length(blocal)
      b[mindexset.lid_to_gid[i]] += blocal[i]
    end
  end
  b
end



function assemble_global_vector(::Union{Type{RowsComputedLocally{true}},Type{OwnedCellsStrategy{true}}},
                                ::Type{M},
                                b::M,
                                m::DistributedIndexSet) where M <: Vector
  b
end
