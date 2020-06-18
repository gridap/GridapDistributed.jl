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

function allocate_local_vector(
  ::Type{PETSc.Vec{Float64}},
  indices::MPIPETScDistributedIndexSet,
)
  DistributedData(indices) do part,index
   fill(0.0,length(index.lid_to_gid))
  end
end

function Gridap.Algebra.finalize_coo!(
  ::Type{PETSc.Mat{Float64}},IJV::MPIPETScDistributedData,m::MPIPETScDistributedIndexSet,n::MPIPETScDistributedIndexSet)
end

function assemble_global_matrix(
  ::Type{RowsComputedLocally{false}},
  ::Type{PETSc.Mat{Float64}},
  IJV::MPIPETScDistributedData,
  m::MPIPETScDistributedIndexSet,
  n::MPIPETScDistributedIndexSet,
)
  ngrows = num_gids(m)
  ngcols = num_gids(n)
  nlrows = num_owned_entries(m)
  nlcols = nlrows

  I, J, V = IJV.part
  ncols_Alocal = maximum(J)
  for i = 1:length(I)
    I[i] = m.app_to_petsc_locidx[I[i]]
  end
  Alocal = sparse_from_coo(
    Gridap.Algebra.SparseMatrixCSR{0,Float64,Int64},
    I,
    J,
    V,
    nlrows,
    ncols_Alocal,
  )
  for i = 1:length(Alocal.colval)
    Alocal.colval[i] = m.lid_to_gid_petsc[Alocal.colval[i]+1] - 1
  end

  p = Ref{PETSc.C.Mat{Float64}}()
  f(buf)= if isempty(buf)
      Ptr{PETSc.C.PetscInt}(0)
    else
      isa(buf,Vector{PETSc.C.PetscInt}) ? buf : PETSc.C.PetscInt[ i for i in buf ]
    end

  PETSc.C.chk(PETSc.C.MatCreateMPIAIJWithArrays(
    get_comm(m).comm,
    nlrows,
    nlcols,
    ngrows,
    ngcols,
    f(Alocal.rowptr),
    f(Alocal.colval),
    Alocal.nzval,
    p,
  ))
  A=PETSc.Mat(p[])
end

function assemble_global_matrix(
  ::Type{OwnedCellsStrategy{false}},
  ::Type{PETSc.Mat{Float64}},
  IJV::MPIPETScDistributedData,
  m::MPIPETScDistributedIndexSet,
  n::MPIPETScDistributedIndexSet,
)
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
  I,J,V = IJV.part
  for (i,j,v) in zip(I,J,V)
    A[m.lid_to_gid_petsc[i],n.lid_to_gid_petsc[j]]=v
  end

  PETSc.AssemblyBegin(A, PETSc.C.MAT_FINAL_ASSEMBLY)
  PETSc.AssemblyEnd(A, PETSc.C.MAT_FINAL_ASSEMBLY)
  A
end

function assemble_global_vector(
  ::Type{OwnedCellsStrategy{false}},
  ::Type{PETSc.Vec{Float64}},
  db::MPIPETScDistributedData,
  indices::MPIPETScDistributedIndexSet)
  vec = allocate_vector(PETSc.Vec{Float64},indices)
  PETSc.setindex0!(vec, db.part, indices.lid_to_gid_petsc .- 1)
  PETSc.AssemblyBegin(vec)
  PETSc.AssemblyEnd(vec)
  vec
end

function assemble_global_vector(
  ::Type{RowsComputedLocally{false}},
  ::Type{PETSc.Vec{Float64}},
  db::MPIPETScDistributedData,
  indices::MPIPETScDistributedIndexSet)
  vec = allocate_vector(PETSc.Vec{Float64},indices)

  part = MPI.Comm_rank(get_comm(indices).comm)+1
  owned_pos = (indices.parts.part.lid_to_owner .== part)
  bowned    = db.part[owned_pos]
  l2g_petsc = indices.lid_to_gid_petsc[owned_pos] .- 1

  PETSc.setindex0!(vec, bowned, l2g_petsc)
  PETSc.AssemblyBegin(vec)
  PETSc.AssemblyEnd(vec)
  vec
end
