function test_assemble(comm,model,U,V,das)
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, das)
  trian=Triangulation(das,model)
  degree=2
  dΩ=Measure(trian,degree)
  v=get_fe_basis(V)
  u=get_trial_fe_basis(U)
  veccont = ∫(1*v)dΩ
  matcont = ∫(v*u)dΩ
  tol=1.0e-14
  matdata=collect_cell_matrix(U,V,matcont)
  A1 = assemble_matrix(assem,matdata)
  vecdata=collect_cell_vector(V,veccont)
  b1 = assemble_vector(assem,vecdata)
  @test sum(b1)-1 < tol
  if isa(comm,SequentialCommunicator)
    @test ones(1,size(A1,1))*A1*ones(size(A1,2)) ≈ [1]
  end
  data = collect_cell_matrix_and_vector(U,V,matcont,veccont)
  A2,b2 = assemble_matrix_and_vector(assem,data)
  @test sum(b2)-1 < tol
  if isa(comm,SequentialCommunicator)
    @test ones(1,size(A2,1))*A2*ones(size(A2,2)) ≈ [1]
    @test norm(A1-A2) ≈ 0.0
  end
  @test norm(b1-b2) < tol
end

function test_allocate_assemble_add(comm,model,U,V,das)
  assem = SparseMatrixAssembler(matrix_type, vector_type, U, V, das)
  trian=Triangulation(das,model)
  degree=2
  dΩ=Measure(trian,degree)
  v=get_fe_basis(V)
  u=get_trial_fe_basis(U)
  veccont = ∫(1*v)dΩ
  matcont = ∫(v*u)dΩ

  tol=1.0e-14
  matdata=collect_cell_matrix(U,V,matcont)
  A1 = assemble_matrix(assem,matdata)
  vecdata=collect_cell_vector(V,veccont)
  b1 = assemble_vector(assem,vecdata)
  @test sum(b1) ≈ 1
  if isa(comm,SequentialCommunicator)
    @test ones(1,size(A1,1))*A1*ones(size(A1,2)) ≈ [1]
  end
  A3 = allocate_matrix(assem,matdata)
  if isa(comm,SequentialCommunicator)
     @test findnz(A1)[1] == findnz(A3)[1]
     @test findnz(A1)[2] == findnz(A3)[2]
  end
  assemble_matrix!(A3,assem,matdata)
  if isa(comm,SequentialCommunicator)
    @test norm(A1-A3) ≈ 0.0
  end
  assemble_matrix_add!(A3,assem,matdata)
  if isa(comm,SequentialCommunicator)
    @test norm(2*A1-A3) ≈ 0.0
  end
  b3 = allocate_vector(assem,vecdata)
  assemble_vector!(b3,assem,vecdata)
  @test norm(b3-b1) < tol
  assemble_vector_add!(b3,assem,vecdata)
  @test norm(b3-2*b1) < tol
  data = collect_cell_matrix_and_vector(U,V,matcont,veccont)
  A4, b4 = allocate_matrix_and_vector(assem,data)
  assemble_matrix_and_vector!(A4,b4,assem,data)
  if isa(comm,SequentialCommunicator)
    @test norm(A1-A4) ≈ 0.0
  end
  @test norm(b1-b4) < tol
  assemble_matrix_and_vector_add!(A4,b4,assem,data)
  @test norm(b4-2*b1) < tol
end
