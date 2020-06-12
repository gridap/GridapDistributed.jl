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
