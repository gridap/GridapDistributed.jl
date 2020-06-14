function Gridap.Algebra.allocate_coo_vectors(
  ::Type{PETSc.Mat{Float64}},
  dn::MPIPETScDistributedData,
)
  DistributedData(dn) do part, n
    I = Vector{Int}(undef, n)
    J = Vector{Int}(undef, n)
    V = Vector{Float64}(undef, n)
    (I, J, V)
  end
end

function Gridap.Algebra.finalize_coo!(
  ::Type{PETSc.Mat{Float64}},IJV::MPIPETScDistributedData,m::MPIPETScDistributedIndexSet,n::MPIPETScDistributedIndexSet)
end

function Gridap.Algebra.sparse_from_coo(
  ::Type{PETSc.Mat{Float64}},IJV::MPIPETScDistributedData,m::MPIPETScDistributedIndexSet,n::MPIPETScDistributedIndexSet)
  ngrows = num_gids(m)
  ngcols = num_gids(n)
  nlrows = num_owned_entries(m)
  nlcols = nlrows

  # TO-DO: Properly manage PREALLOCATION building the sparsity pattern of the matrix
  A=Mat(Float64, ngrows, ngcols;
        mlocal=nlrows, nlocal=nlcols, nz=100, onz=100,
        comm=get_comm(m).comm, mtype=PETSc.C.MATMPIAIJ)

  A.insertmode = PETSc.C.ADD_VALUES

  # TO-DO: Not efficient!, one call to PETSc function per each injection
  do_on_parts(get_comm(m),IJV) do part, IJV
     I,J,V = IJV
     for (i,j,v) in zip(I,J,V)
       A[i,j]=v
     end
  end

  PETSc.AssemblyBegin(A, PETSc.C.MAT_FINAL_ASSEMBLY)
  PETSc.AssemblyEnd(A, PETSc.C.MAT_FINAL_ASSEMBLY)

  A

end

@noinline function Gridap.FESpaces._assemble_matrix_and_vector_fill!(
  ::Type{M},nini,I,J,V,b,vals_cache,rows_cache,cols_cache,cell_vals,cell_rows,cell_cols,strategy) where M <: SparseMatrixCSR
  n = nini
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    matvals, vecvals = vals
    for (j,gidcol) in enumerate(cols)
      if gidcol > 0 && col_mask(strategy,gidcol)
        _gidcol = col_map(strategy,gidcol)
        for (i,gidrow) in enumerate(rows)
          if gidrow > 0 && row_mask(strategy,gidrow)
            _gidrow = row_map(strategy,gidrow)
            if is_entry_stored(M,gidrow,gidcol)
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
      if gidrow > 0 && row_mask(strategy,gidrow)
        _gidrow = row_map(strategy,gidrow)
        # TO-DO!!!
        #bi = vecvals[i]
        #b[_gidrow] += bi
        b[_gidrow] = vecvals[i]
      end
    end
  end
  n
end
