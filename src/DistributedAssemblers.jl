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
  u::DistributedFESpace,v::DistributedFESpace,mterms,vterms,uhd::FEFunction)
  DistributedData(u,v,mterms,vterms,uhd) do part, (u,_), (v,_), mterms, vterms, uhd
    collect_cell_matrix_and_vector(u,v,mterms,vterms,uhd)
  end
end


struct DistributedAssemblyStrategy{T<:AssemblyStrategy}
  strategies::DistributedData{T}
end

function get_distributed_data(dstrategy::DistributedAssemblyStrategy)
  dstrategy.strategies
end

struct DistributedAssembler{M,V,AS} <: Assembler
  matrix_type             :: Type{M}
  vector_type             :: Type{V}
  trial::DistributedFESpace
  test::DistributedFESpace
  assems::DistributedData{<:Assembler}
  strategy::DistributedAssemblyStrategy{AS}
end

struct ArtificeSparseMatrixToLeverageDefaults{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti} end

function Gridap.Algebra.is_entry_stored(::Type{<:ArtificeSparseMatrixToLeverageDefaults},i,j)
  true
end

"""
    allocate_local_vector(::DistributedAssemblyStrategy, ::Type{V}, indices) where {V}

Allocate the local vector required in order to assemble a global vector
of type V accordingly to the assembly algorithm underlying the assembly strategy.  The global vector is indexable using indices.
"""
function allocate_local_vector(::DistributedAssemblyStrategy, ::Type{V}, indices::DistributedIndexSet) where {V}
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

function assemble_global_matrix(::DistributedAssemblyStrategy, ::Type{M},
                                ::DistributedData,
                                ::DistributedIndexSet,
                                ::DistributedIndexSet) where {M}
   @abstractmethod
end

function assemble_global_vector(::DistributedAssemblyStrategy, ::Type{V},
                                ::DistributedData,
                                ::DistributedIndexSet) where {V}
   @abstractmethod
end

function get_distributed_data(dassem::DistributedAssembler)
  dassem.assems
end

function Gridap.FESpaces.get_assembly_strategy(a::DistributedAssembler)
  a.strategy
end

function Gridap.FESpaces.allocate_matrix(dassem::DistributedAssembler,dmatdata)
  dm1 = DistributedData(dassem,dmatdata) do part, assem, matdata
    builder=SparseMatrixBuilder(ArtificeSparseMatrixToLeverageDefaults{Float64,Int64},MinMemory())
    m1=nz_counter(builder,(get_rows(assem),get_cols(assem)))
    symbolic_loop_matrix!(m1,assem,matdata)
  end

  dm2 = DistributedData(dm1,dassem,dmatdata) do part, m1, assem, matdata
    m2=nz_allocation(m1)
    symbolic_loop_matrix!(m2,assem,matdata)
    m2
  end

  dIJV = DistributedData(dm2) do part, m2
    m2.I,m2.J,m2.V
  end

  finalize_coo!(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)
  A = assemble_global_matrix(dassem.strategy,
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
  dm1 = DistributedData(dassem,ddata) do part, assem, data
    mbuilder=SparseMatrixBuilder(ArtificeSparseMatrixToLeverageDefaults{Float64,Int64},MinMemory())
    m1=nz_counter(mbuilder,(get_rows(assem),get_cols(assem)))
    vbuilder=ArrayBuilder(get_local_vector_type(dassem.vector_type))
    v1=nz_counter(vbuilder,(get_rows(assem),))
    symbolic_loop_matrix_and_vector!(m1,v1,assem,data)
    m1
  end

  gids = dassem.test.gids
  db = allocate_local_vector(dassem.strategy,dassem.vector_type,gids)

  dm2 = DistributedData(dm1,db,dassem,ddata) do part, m1, b, assem, data
    m2=nz_allocation(m1)
    symbolic_loop_matrix_and_vector!(m2,b,assem,data)
    m2
  end

  dIJV = DistributedData(dm2) do part, m2
    m2.I,m2.J,m2.V
  end

  finalize_coo!(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)
  A = assemble_global_matrix(dassem.strategy,
                             dassem.matrix_type,
                             dIJV,
                             dassem.test.gids,
                             dassem.trial.gids)

  b = assemble_global_vector(dassem.strategy,
                             dassem.vector_type,
                             db,
                             dassem.test.gids)
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
  dm1 = DistributedData(dassem,dmatdata) do part, assem, matdata
    builder=SparseMatrixBuilder(ArtificeSparseMatrixToLeverageDefaults{Float64,Int64},MinMemory())
    m1=nz_counter(builder,(get_rows(assem),get_cols(assem)))
    symbolic_loop_matrix!(m1,assem,matdata)
  end

  dm2 = DistributedData(dm1,dassem,dmatdata) do part, m1, assem, matdata
    m2=nz_allocation(m1)
    numeric_loop_matrix!(m2,assem,matdata)
    m2
  end

  dIJV = DistributedData(dm2) do part, m2
    m2.I,m2.J,m2.V
  end

  finalize_coo!(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)
  A = assemble_global_matrix(dassem.strategy,
                             dassem.matrix_type,
                             dIJV,
                             dassem.test.gids,
                             dassem.trial.gids)
end

function Gridap.FESpaces.assemble_vector(dassem::DistributedAssembler, dvecdata)
  gids = dassem.test.gids
  db = allocate_local_vector(dassem.strategy,dassem.vector_type,gids)
  do_on_parts(dassem,dvecdata,db) do part, assem, vecdata, b
     numeric_loop_vector!(b,assem,vecdata)
  end
  b = assemble_global_vector(dassem.strategy,
                             dassem.vector_type,
                             db,
                             dassem.test.gids)
end

function Gridap.FESpaces.assemble_matrix_and_vector(dassem::DistributedAssembler,ddata)
  dm1 = DistributedData(dassem,ddata) do part, assem, data
    mbuilder=SparseMatrixBuilder(ArtificeSparseMatrixToLeverageDefaults{Float64,Int64},MinMemory())
    m1=nz_counter(mbuilder,(get_rows(assem),get_cols(assem)))
    vbuilder=ArrayBuilder(get_local_vector_type(dassem.vector_type))
    v1=nz_counter(vbuilder,(get_rows(assem),))
    symbolic_loop_matrix_and_vector!(m1,v1,assem,data)
    m1
  end

  gids = dassem.test.gids
  db = allocate_local_vector(dassem.strategy,dassem.vector_type,gids)

  dm2 = DistributedData(dm1,db,dassem,ddata) do part, m1, b, assem, data
    m2=nz_allocation(m1)
    numeric_loop_matrix_and_vector!(m2,b,assem,data)
    m2
  end

  dIJV = DistributedData(dm2) do part, m2
    m2.I,m2.J,m2.V
  end

  finalize_coo!(dassem.matrix_type,dIJV,dassem.test.gids,dassem.trial.gids)
  A = assemble_global_matrix(dassem.strategy,
                             dassem.matrix_type,
                             dIJV,
                             dassem.test.gids,
                             dassem.trial.gids)

  b = assemble_global_vector(dassem.strategy,
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
  dstrategy::DistributedAssemblyStrategy)

  assems = DistributedData(
    dtrial.spaces,dtest.spaces,dstrategy) do part, U, V, strategy
    SparseMatrixAssembler(get_local_matrix_type(matrix_type),
                          get_local_vector_type(vector_type),
                          U,
                          V,
                          strategy)
  end

  DistributedAssembler(matrix_type,
                       vector_type,
                       dtrial,
                       dtest,
                       assems,
                       dstrategy)
end
