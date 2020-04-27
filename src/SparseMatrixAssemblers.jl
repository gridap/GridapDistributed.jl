
# TODO @fverdugo 
# This file is a copy of the corresponding one in Gridap
# Some extension of the assembler is needed for parallel computations
# for the moment, we work on this extension here, but in the future these changes
# will be ported to Gridap in order to avoid to repeat so many lines of code.
# The same will be needed for multi-field since the multifield case has its own assembler.
# Perhaps a good time to unify single-field and multifield assemblers
#

# This will allow to consider different assembly strategies in the same framework

abstract type AssemblyStrategy end

function row_map(a::AssemblyStrategy,row)
  @abstractmethod
end

function col_map(a::AssemblyStrategy,col)
  @abstractmethod
end

function row_mask(a::AssemblyStrategy,cell,row)
  @abstractmethod
end

function col_mask(a::AssemblyStrategy,cell,col)
  @abstractmethod
end

# This is one of the usual assembly strategies in parallel FE computations
# (but not the one we have used in the parallel agfem paper)
# Each proc owns a set of matrix / vector rows (and all cols in these rows)
# Each proc computes locally all values in the owned rows
# This typically requires to loop also over ghost cells
struct RowsComputedLocally <: AssemblyStrategy
  lid_to_gid::Vector{Int}
  cell_to_owner::Vector{Int}
  lid_to_owner::Vector{Int}
end

function row_map(a::RowsComputedLocally,row)
  a.lid_to_gid[row]
end

function col_map(a::RowsComputedLocally,col)
  a.lid_to_gid[col]
end

function row_mask(a::RowsComputedLocally,cell,row)
  a.cell_to_owner[cell] == a.lid_to_owner[row]
end

function col_mask(a::RowsComputedLocally,cell,col)
  true
end

struct SparseMatrixAssemblerX <: Assembler
  trial::SingleFieldFESpace
  test::SingleFieldFESpace
  strategy::AssemblyStrategy
end

Gridap.FESpaces.get_test(a::SparseMatrixAssemblerX) = a.test

Gridap.FESpaces.get_trial(a::SparseMatrixAssemblerX) = a.trial

function Gridap.FESpaces.allocate_vector(a::SparseMatrixAssemblerX,term_to_cellidsrows)
  allocate_vector(a.vector_type,a.test)
end

function Gridap.FESpaces.assemble_vector!(b,a::SparseMatrixAssemblerX,term_to_cellvec,term_to_cellidsrows)
  celldofs = get_cell_dofs(a.test)
  fill!(b,zero(eltype(b)))
  for (cellvec, cellids) in zip(term_to_cellvec,term_to_cellidsrows)
    rows = reindex(celldofs,cellids)
    vals = apply_constraints_vector(a.test,cellvec,cellids)
    rows_cache = array_cache(rows)
    vals_cache = array_cache(vals)
    _assemble_vector!(b,vals_cache,rows_cache,vals,rows,a.strategy)
  end
  b
end

function _assemble_vector!(vec,vals_cache,rows_cache,cell_vals,cell_rows,strategy)
  @assert length(cell_vals) == length(cell_rows)
  for cell in 1:length(cell_rows)
    rows = getindex!(rows_cache,cell_rows,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    for (i,gid) in enumerate(rows)
      if gid > 0 && row_mask(strategy,cell,gid)
        _gid = row_map(strategy,gid)
        add_entry!(vec,vals[i],_gid)
      end
    end
  end
end

function Gridap.FESpaces.allocate_matrix(a::SparseMatrixAssemblerX,term_to_cellidsrows, term_to_cellidscols)
  celldofs_rows = get_cell_dofs(a.test)
  celldofs_cols = get_cell_dofs(a.trial)
  n = 0
  for (cellidsrows,cellidscols) in zip(term_to_cellidsrows,term_to_cellidscols)
    cell_rows = reindex(celldofs_rows,cellidsrows)
    cell_cols = reindex(celldofs_cols,cellidscols)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    @assert length(cell_cols) == length(cell_rows)
    n += _count_matrix_entries(a.matrix_type,rows_cache,cols_cache,cell_rows,cell_cols,a.dofmap,a.cellmask,a.strategy)
  end
  I, J, V = allocate_coo_vectors(a.matrix_type,n)
  nini = 0
  for (cellidsrows,cellidscols) in zip(term_to_cellidsrows,term_to_cellidscols)
    cell_rows = reindex(celldofs_rows,cellidsrows)
    cell_cols = reindex(celldofs_cols,cellidscols)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    nini = _allocate_matrix!(a.matrix_type,nini,I,J,rows_cache,cols_cache,cell_rows,cell_cols,a.strategy)
  end
  finalize_coo!(a.matrix_type,I,J,V,a.test,a.trial)
  sparse_from_coo(a.matrix_type,I,J,V,a.test,a.trial)
end

@noinline function _count_matrix_entries(::Type{M},rows_cache,cols_cache,cell_rows,cell_cols,strategy) where M
  n = 0
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    for gidcol in cols
      if gidcol > 0 && col_mask(strategy,cell,gidcol)
        _gidcol = col_map(strategy,gidcol)
        for gidrow in rows
          if gidrow > 0 && row_mask(strategy,cell,gidrow)
            _gidrow = row_map(strategy,gidrow)
            if is_entry_stored(M,_gidrow,_gidcol)
              n += 1
            end
          end
        end
      end
    end
  end
  n
end

@noinline function _allocate_matrix!(a::Type{M},nini,I,J,rows_cache,cols_cache,cell_rows,cell_cols,strategy) where M
  n = nini
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    for gidcol in cols
      if gidcol > 0 && col_mask(strategy,cell,gidcol)
        _gidcol = col_map(strategy,gidcol)
        for gidrow in rows
          if gidrow > 0 && row_mask(strategy,cell,gidrow)
            _gidrow = row_map(strategy,gidrow)
            if is_entry_stored(M,_gidrow,_gidcol)
              n += 1
              @inbounds I[n] = _gidrow
              @inbounds J[n] = _gidcol
            end
          end
        end
      end
    end
  end
  n
end

function Gridap.FESpaces.assemble_matrix!(
  mat,a::SparseMatrixAssemblerX, term_to_cellmat, term_to_cellidsrows, term_to_cellidscols)
  z = zero(eltype(mat))
  fill_entries!(mat,z)
  assemble_matrix_add!(mat,a,term_to_cellmat,term_to_cellidsrows,term_to_cellidscols)
end

function Gridap.FESpaces.assemble_matrix_add!(
  mat,a::SparseMatrixAssemblerX, term_to_cellmat, term_to_cellidsrows, term_to_cellidscols)
  celldofs_rows = get_cell_dofs(a.test)
  celldofs_cols = get_cell_dofs(a.trial)
  for (cellmat_rc,cellidsrows,cellidscols) in zip(term_to_cellmat,term_to_cellidsrows,term_to_cellidscols)
    cell_rows = reindex(celldofs_rows,cellidsrows)
    cell_cols = reindex(celldofs_cols,cellidscols)
    cellmat_r = apply_constraints_matrix_cols(a.trial,cellmat_rc,cellidscols)
    cellmat = apply_constraints_matrix_rows(a.test,cellmat_r,cellidsrows)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    vals_cache = array_cache(cellmat)
    _assemble_matrix!(mat,vals_cache,rows_cache,cols_cache,cellmat,cell_rows,cell_cols,a.strategy)
  end
  mat
end

function _assemble_matrix!(mat,vals_cache,rows_cache,cols_cache,cell_vals,cell_rows,cell_cols,strategy)
  @assert length(cell_cols) == length(cell_rows)
  @assert length(cell_vals) == length(cell_rows)
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    for (j,gidcol) in enumerate(cols)
      if gidcol > 0 && col_mask(strategy,cell,gidcol)
        _gidcol = col_map(strategy,gidcol)
        for (i,gidrow) in enumerate(rows)
          if gidrow > 0 && row_mask(strategy,cell,gidrow)
            _gidrow = row_map(strategy,gidrow)
            v = vals[i,j]
            add_entry!(mat,v,_gidrow,_gidcol)
          end
        end
      end
    end
  end
end

function Gridap.FESpaces.assemble_matrix(
  a::SparseMatrixAssemblerX, term_to_cellmat, term_to_cellidsrows, term_to_cellidscols)
  celldofs_rows = get_cell_dofs(a.test)
  celldofs_cols = get_cell_dofs(a.trial)
  n = 0
  for (cellidsrows,cellidscols) in zip(term_to_cellidsrows,term_to_cellidscols)
    cell_rows = reindex(celldofs_rows,cellidsrows)
    cell_cols = reindex(celldofs_cols,cellidscols)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    @assert length(cell_cols) == length(cell_rows)
    n += _count_matrix_entries(a.matrix_type,rows_cache,cols_cache,cell_rows,cell_cols,a.strategy)
  end
  I, J, V = allocate_coo_vectors(a.matrix_type,n)
  nini = 0
  for (cellmat_rc,cellidsrows,cellidscols) in zip(term_to_cellmat,term_to_cellidsrows,term_to_cellidscols)
    cell_rows = reindex(celldofs_rows,cellidsrows)
    cell_cols = reindex(celldofs_cols,cellidscols)
    cellmat_r = apply_constraints_matrix_cols(a.trial,cellmat_rc,cellidscols)
    cellmat = apply_constraints_matrix_rows(a.test,cellmat_r,cellidsrows)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    vals_cache = array_cache(cellmat)
    @assert length(cell_cols) == length(cell_rows)
    @assert length(cellmat) == length(cell_rows)
    nini = _assemble_matrix_fill!(a.matrix_type,nini,I,J,V,vals_cache,rows_cache,cols_cache,cellmat,cell_rows,cell_cols,a.strategy)
  end
  finalize_coo!(a.matrix_type,I,J,V,a.test,a.trial)
  sparse_from_coo(a.matrix_type,I,J,V,a.test,a.trial)
end

@noinline function _assemble_matrix_fill!(::Type{M},nini,I,J,V,vals_cache,rows_cache,cols_cache,cell_vals,cell_rows,cell_cols,strategy) where M
  n = nini
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    for (j,gidcol) in enumerate(cols)
      if gidcol > 0 && col_mask(strategy,cell,gidcol)
        _gidcol = col_map(strategy,gidcol)
        for (i,gidrow) in enumerate(rows)
          if gidrow > 0 && row_mask(strategy,cell,gidrow)
            _gidrow = row_map(strategy,gidrow)
            if is_entry_stored(M,_gidrow,_gidcol)
              n += 1
              @inbounds v = vals[i,j]
              @inbounds I[n] = _gidrow
              @inbounds J[n] = _gidcol
              @inbounds V[n] = v
            end
          end
        end
      end
    end
  end
  n
end

function Gridap.FESpaces.assemble_matrix_and_vector!(A,b,a::SparseMatrixAssemblerX, matvecdata, matdata, vecdata)
  z = zero(eltype(A))
  fill_entries!(A,z)
  fill!(b,zero(eltype(b)))
  celldofs_rows = get_cell_dofs(a.test)
  celldofs_cols = get_cell_dofs(a.trial)

  for (cellmatvec_rc,cellidsrows,cellidscols) in zip(matvecdata...)
    cell_rows = reindex(celldofs_rows,cellidsrows)
    cell_cols = reindex(celldofs_cols,cellidscols)
    cellmatvec_r = apply_constraints_matrix_and_vector_cols(a.trial,cellmatvec_rc,cellidscols)
    cellmatvec = apply_constraints_matrix_and_vector_rows(a.test,cellmatvec_r,cellidsrows)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    vals_cache = array_cache(cellmatvec)
    _assemble_matrix_and_vector!(A,b,vals_cache,rows_cache,cols_cache,cellmatvec,cell_rows,cell_cols,a.strategy)
  end

  for (cellmat_rc,cellidsrows,cellidscols) in zip(matdata...)
    cell_rows = reindex(celldofs_rows,cellidsrows)
    cell_cols = reindex(celldofs_cols,cellidscols)
    cellmat_r = apply_constraints_matrix_cols(a.trial,cellmat_rc,cellidscols)
    cellmat = apply_constraints_matrix_rows(a.test,cellmat_r,cellidsrows)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    vals_cache = array_cache(cellmat)
    _assemble_matrix!(A,vals_cache,rows_cache,cols_cache,cellmat,cell_rows,cell_cols,a.strategy)
  end

  for (cellvec, cellids) in zip(vecdata...)
    rows = reindex(celldofs_rows,cellids)
    vals = apply_constraints_vector(a.test,cellvec,cellids)
    rows_cache = array_cache(rows)
    vals_cache = array_cache(vals)
    _assemble_vector!(b,vals_cache,rows_cache,vals,rows)
  end

  A, b
end

function _assemble_matrix_and_vector!(A,b,vals_cache,rows_cache,cols_cache,cell_vals,cell_rows,cell_cols,strategy)
  @assert length(cell_cols) == length(cell_rows)
  @assert length(cell_vals) == length(cell_rows)
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    matvals, vecvals = vals
    for (j,gidcol) in enumerate(cols)
      if gidcol > 0 && col_mask(strategy,cell,gidcol)
        _gidcol = col_map(strategy,gidcol)
        for (i,gidrow) in enumerate(rows)
          if gidrow > 0 && row_mask(strategy,cell,gidrow)
            _gidrow = row_map(strategy,gidrow)
            v = matvals[i,j]
            add_entry!(A,v,_gidrow,_gidcol)
          end
        end
      end
    end
    for (i,gidrow) in enumerate(rows)
      if gidrow > 0 && row_mask(strategy,cell,gidrow)
        _gidrow = row_map(strategy,gidrow)
        bi = vecvals[i]
        add_entry!(b,bi,_gidrow)
      end
    end
  end
end

function Gridap.FESpaces.assemble_matrix_and_vector( a::SparseMatrixAssemblerX, matvecdata, matdata, vecdata)
  celldofs_rows = get_cell_dofs(a.test)
  celldofs_cols = get_cell_dofs(a.trial)

  term_to_cellidsrows, term_to_cellidscols,  =  _rearange_cell_ids(matvecdata,matdata,vecdata)

  n = 0
  for (cellidsrows,cellidscols) in zip(term_to_cellidsrows,term_to_cellidscols)
    cell_rows = reindex(celldofs_rows,cellidsrows)
    cell_cols = reindex(celldofs_cols,cellidscols)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    @assert length(cell_cols) == length(cell_rows)
    n += _count_matrix_entries(a.matrix_type,rows_cache,cols_cache,cell_rows,cell_cols,a.strategy)
  end

  I, J, V = allocate_coo_vectors(a.matrix_type,n)
  b = zero_free_values(a.test)
  nini = 0

  for (cellmatvec_rc,cellidsrows,cellidscols) in zip(matvecdata...)
    cell_rows = reindex(celldofs_rows,cellidsrows)
    cell_cols = reindex(celldofs_cols,cellidscols)
    cellmatvec_r = apply_constraints_matrix_and_vector_cols(a.trial,cellmatvec_rc,cellidscols)
    cellmatvec = apply_constraints_matrix_and_vector_rows(a.test,cellmatvec_r,cellidsrows)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    vals_cache = array_cache(cellmatvec)
    @assert length(cell_cols) == length(cell_rows)
    @assert length(cellmatvec) == length(cell_rows)
    nini = _assemble_matrix_and_vector_fill!(a.matrix_type,nini,I,J,V,b,vals_cache,rows_cache,cols_cache,cellmatvec,cell_rows,cell_cols,a.strategy)
  end

  for (cellmat_rc,cellidsrows,cellidscols) in zip(matdata...)
    cell_rows = reindex(celldofs_rows,cellidsrows)
    cell_cols = reindex(celldofs_cols,cellidscols)
    cellmat_r = apply_constraints_matrix_cols(a.trial,cellmat_rc,cellidscols)
    cellmat = apply_constraints_matrix_rows(a.test,cellmat_r,cellidsrows)
    rows_cache = array_cache(cell_rows)
    cols_cache = array_cache(cell_cols)
    vals_cache = array_cache(cellmat)
    @assert length(cell_cols) == length(cell_rows)
    @assert length(cellmat) == length(cell_rows)
    nini = _assemble_matrix_fill!(a.matrix_type,nini,I,J,V,vals_cache,rows_cache,cols_cache,cellmat,cell_rows,cell_cols,a.strategy)
  end

  for (cellvec, cellids) in zip(vecdata...)
    rows = reindex(celldofs_rows,cellids)
    vals = apply_constraints_vector(a.test,cellvec,cellids)
    rows_cache = array_cache(rows)
    vals_cache = array_cache(vals)
    _assemble_vector!(b,vals_cache,rows_cache,vals,rows,a.strategy)
  end

  finalize_coo!(a.matrix_type,I,J,V,a.test,a.trial)
  A = sparse_from_coo(a.matrix_type,I,J,V,a.test,a.trial)

  (A, b)
end

@noinline function _assemble_matrix_and_vector_fill!(::Type{M},nini,I,J,V,b,vals_cache,rows_cache,cols_cache,cell_vals,cell_rows,cell_cols,strategy) where M
  n = nini
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    matvals, vecvals = vals
    for (j,gidcol) in enumerate(cols)
      if gidcol > 0 && col_mask(strategy,cell,gidcol)
        _gidcol = col_map(strategy,gidcol)
        for (i,gidrow) in enumerate(rows)
          if gidrow > 0 && row_mask(strategy,cell,gidrow)
            _gidrow = row_map(strategy,gidrow)
            if is_entry_stored(M,_gidrow,_gidcol)
              n += 1
              @inbounds v = matvals[i,j]
              @inbounds I[n] = _gidrow
              @inbounds J[n] = _gidcol
              @inbounds V[n] = v
            end
          end
        end
      end
    end
    for (i,gidrow) in enumerate(rows)
      if gidrow > 0 && row_mask(strategy,cell,gidrow)
        _gidrow = row_map(strategy,gidrow)
      bi = vecvals[i]
      add_entry!(b,bi,_gidrow)
      end
    end
  end
  n
end
