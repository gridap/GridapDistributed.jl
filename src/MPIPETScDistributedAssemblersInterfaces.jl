
"""
    allocate_vector(::Type{V},indices) where V

Allocate a vector of type `V` indexable at the indices `indices`
"""
function Gridap.Algebra.allocate_vector(
  ::Type{GridapDistributedPETScWrappers.Vec{Float64}},
  indices::MPIPETScDistributedIndexSet,
)
  ng = num_gids(indices)
  nl = num_owned_entries(indices)
  vec=GridapDistributedPETScWrappers.Vec(Float64, ng; mlocal = nl, comm = get_comm(indices).comm)
  vec.insertmode = GridapDistributedPETScWrappers.C.ADD_VALUES
  GridapDistributedPETScWrappers.set_local_to_global_mapping(vec,indices.IS)
  vec
end

function Gridap.Algebra.fill_entries!(a::GridapDistributedPETScWrappers.Mat{Float64},v::Number)
  fill!(a,v)
  a
end

function Gridap.FESpaces.assemble_matrix_and_vector_add!(dmat::GridapDistributedPETScWrappers.Mat{Float64},dvec::GridapDistributedPETScWrappers.Vec{Float64},dassem::DistributedAssembler, ddata)
  do_on_parts(dassem,ddata,dmat,dvec) do part, assem, data, mat, vec
    assemble_matrix_and_vector_add!(mat,vec,assem,data)
  end
  GridapDistributedPETScWrappers.assemble(dmat)
  GridapDistributedPETScWrappers.assemble(dvec)
end

function Gridap.FESpaces.assemble_matrix_add!(dmat::GridapDistributedPETScWrappers.Mat{Float64},dassem::DistributedAssembler, ddata)
  do_on_parts(dassem,ddata,dmat) do part, assem, data, mat
    assemble_matrix_add!(mat,assem,data)
  end
  GridapDistributedPETScWrappers.assemble(dmat)
end

function Gridap.FESpaces.assemble_vector_add!(dvec::GridapDistributedPETScWrappers.Vec{Float64},dassem::DistributedAssembler, ddata)
  do_on_parts(dassem,ddata,dvec) do part, assem, data, vec
    assemble_vector_add!(vec,assem,data)
  end
  GridapDistributedPETScWrappers.assemble(dvec)
end



function get_local_vector_type(::Type{GridapDistributedPETScWrappers.Vec{Float64}})
  Vector{Float64}
end

function get_local_matrix_type(::Type{GridapDistributedPETScWrappers.Mat{Float64}})
  SparseMatrixCSR{1,Float64,Int64}
end

function allocate_local_vector(
  strat::Union{DistributedAssemblyStrategy{OwnedAndGhostCellsAssemblyStrategy{false}},DistributedAssemblyStrategy{OwnedCellsAssemblyStrategy{false}}},
  ::Type{GridapDistributedPETScWrappers.Vec{Float64}},
  indices::MPIPETScDistributedIndexSet,
)
  DistributedData(indices) do part,index
   T = get_local_vector_type(GridapDistributedPETScWrappers.Vec{Float64})
   lvec=T(undef,length(index.lid_to_gid))
   fill!(lvec,zero(eltype(T)))
   lvec
  end
end

function Gridap.Algebra.finalize_coo!(
  ::Type{GridapDistributedPETScWrappers.Mat{Float64}},IJV::MPIPETScDistributedData,m::MPIPETScDistributedIndexSet,n::MPIPETScDistributedIndexSet)
end

function _convert_buf_to_petscint(buf)
  if isempty(buf)
    Ptr{GridapDistributedPETScWrappers.C.PetscInt}(0)
  else
    isa(buf,Vector{GridapDistributedPETScWrappers.C.PetscInt}) ? buf : GridapDistributedPETScWrappers.C.PetscInt[ i for i in buf ]
  end
end

function assemble_global_matrix(
  strat::DistributedAssemblyStrategy{OwnedAndGhostCellsAssemblyStrategy{false}},
  ::Type{GridapDistributedPETScWrappers.Mat{Float64}},
  IJV::MPIPETScDistributedData,
  m::MPIPETScDistributedIndexSet,
  n::MPIPETScDistributedIndexSet,
)
  I, J, V = IJV.part
  for i = 1:length(I)
    I[i] = m.app_to_petsc_locidx[I[i]]
  end
  # Build fully assembled local portions
  Alocal =
    build_fully_assembled_local_portion(m,I,J,V,m.lid_to_gid_petsc)

  # Build global matrix from fully assembled local portions
  build_petsc_matrix_from_local_portion(m,n,Alocal)
end

function compute_subdomain_graph_dIS_and_lst_snd(gids, dI)
  # List parts I have to send data to
  function compute_lst_snd(part, gids, I)
    lst_snd = Set{Int}()
    for i = 1:length(I)
        owner = gids.lid_to_owner[I[i]]
        if (owner != part)
          if (!(owner in lst_snd))
            push!(lst_snd, owner)
          end
        end
    end
    collect(lst_snd)
  end

  part_to_lst_snd = DistributedData(compute_lst_snd, gids, dI)
  part_to_num_snd = DistributedData(part_to_lst_snd) do part, lst_snd
    length(lst_snd)
  end

  offsets = gather(part_to_num_snd)
  num_edges = sum(offsets)
  GridapDistributed._fill_offsets!(offsets)
  part_to_offsets = scatter(get_comm(part_to_num_snd), offsets)

  part_to_owned_subdomain_graph_edge_gids =
    DistributedData(part_to_num_snd, part_to_offsets) do part, num_snd, offset
      owned_edge_gids = zeros(Int, num_snd)
      o = offset
      for i = 1:num_snd
        o += 1
        owned_edge_gids[i] = o
      end
      owned_edge_gids
    end

  num_snd    = GridapDistributedPETScWrappers.C.PetscMPIInt(part_to_num_snd.part)
  lst_snd    = convert(Vector{GridapDistributedPETScWrappers.C.PetscMPIInt}, part_to_lst_snd.part) .- GridapDistributedPETScWrappers.C.PetscMPIInt(1)
  snd_buf    = part_to_owned_subdomain_graph_edge_gids.part

  num_rcv    = Ref{GridapDistributedPETScWrappers.C.PetscMPIInt}()
  lst_rcv    = Ref{Ptr{GridapDistributedPETScWrappers.C.PetscMPIInt}}()
  rcv_buf    = Ref{Ptr{Int64}}()

  #GridapDistributedPETScWrappers.C.PetscCommBuildTwoSidedSetType(Float64, get_comm(gids).comm, GridapDistributedPETScWrappers.C.PETSC_BUILDTWOSIDED_ALLREDUCE)
  GridapDistributedPETScWrappers.C.chk(GridapDistributedPETScWrappers.C.PetscCommBuildTwoSided(Float64,
                                 get_comm(gids).comm,
                                 GridapDistributedPETScWrappers.C.PetscMPIInt(1),
                                 MPI.Datatype(Int64).val,
                                 num_snd,
                                 pointer(lst_snd),
                                 Ptr{Cvoid}(pointer(snd_buf)),
                                 num_rcv,
                                 lst_rcv,
                                 Base.unsafe_convert(Ptr{Cvoid},rcv_buf)))


  #TO-DO: Call to PetscFree for lst_rcv and rcv_buf
  #All attempts so far resulted in a SEGfault, so I gave up
  #see https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscCommBuildTwoSided.html
  #for more details

  num_rcv=num_rcv[]
  lst_rcv_vec=unsafe_wrap(Vector{GridapDistributedPETScWrappers.C.PetscMPIInt},lst_rcv[],num_rcv)
  rcv_buf_vec=unsafe_wrap(Vector{Int64},rcv_buf[],num_rcv)

  function compute_subdomain_graph_index_set(part,num_edges,owned_edge_entries)
    lid_to_gid=Vector{Int}(undef,num_snd+num_rcv)
    lid_to_owner=Vector{Int}(undef,num_snd+num_rcv)
    for i=1:num_snd
     lid_to_gid[i]=owned_edge_entries[i]
     lid_to_owner[i]=part
    end
    for i=num_snd+1:num_snd+num_rcv
     lid_to_gid[i]=rcv_buf_vec[i-num_snd]
     lid_to_owner[i]=lst_rcv_vec[i-num_snd]+1
    end
    IndexSet(num_edges, lid_to_gid, lid_to_owner)
  end

  (DistributedIndexSet(compute_subdomain_graph_index_set,
                       get_comm(gids),
                       num_edges,
                       num_edges,
                       part_to_owned_subdomain_graph_edge_gids),
                       part_to_lst_snd)
end

function pack_and_comm_entries(dIS, dIJV, m, n, part_to_lst_snd)
   length_entries =
      DistributedVector(dIS, dIS, m, dIJV, part_to_lst_snd) do part, dIS, gid, IJV, lst_snd
    I,_,_ = IJV
    local_vector = fill(zero(Int),length(dIS.lid_to_owner))
    for i = 1:length(I)
      owner = gid.lid_to_owner[I[i]]
      if (owner != part)
        edge_lid = findfirst((i) -> (i == owner), lst_snd)
        local_vector[edge_lid]+=3
      end
    end
    local_vector
  end
  exchange!(length_entries)
  dd_length_entries = DistributedData(length_entries) do part, length_entries
    collect(length_entries)
  end

  # Pack data to be sent
  exchange_entries_vector =
  DistributedVector(dIS, dIS, dd_length_entries,
                    m, dIJV, part_to_lst_snd) do part, IS, length_entries, test_gids, IJV, lst_snd

    ptrs    = Vector{Int64}(undef, length(length_entries)+1)
    ptrs[1] = 1
    for i=1:length(ptrs)-1
      ptrs[i+1]=ptrs[i]+length_entries[i]
    end
    data = Vector{Float64}(undef, ptrs[end]-1)

    current_pack_position = fill(zero(Int32), length(IS.lid_to_owner))
    I,J,V = IJV
    for i = 1:length(I)
        owner = test_gids.lid_to_owner[I[i]]
        if (owner != part)
          edge_lid = findfirst((i) -> (i == owner), lst_snd)
          row=m.lid_to_gid_petsc[I[i]]
          col=n.lid_to_gid_petsc[J[i]]
          current_pos=current_pack_position[edge_lid]
          data[ptrs[edge_lid]+current_pos]  = row
          data[ptrs[edge_lid]+current_pos+1]= col
          data[ptrs[edge_lid]+current_pos+2]= V[i]
          current_pack_position[edge_lid] += 3
        end
    end
    Table(data,ptrs)
  end
  exchange!(exchange_entries_vector)
  exchange_entries_vector
end

function combine_local_and_remote_entries(dIS, dIJV, m, n, exchange_entries_vector)
  # 3. Combine local + remote entries
  part              = MPI.Comm_rank(get_comm(m).comm)+1
  test_lid_to_owner = m.parts.part.lid_to_owner
  test_lid_to_gid   = m.lid_to_gid_petsc
  test_gid_to_lid   = Dict{Int,Int32}()
  for (lid,gid) in enumerate(test_lid_to_gid)
    test_gid_to_lid[gid] = lid
  end

  trial_lid_to_gid  = n.lid_to_gid_petsc
  trial_gid_to_lid  = Dict{Int,Int32}()
  for (lid,gid) in enumerate(trial_lid_to_gid)
    trial_gid_to_lid[gid] = lid
  end

  #TODO: check with fverdugo if there is an already coded way of
  #      doing this vector pattern operation
  lid_to_owned_lid = fill(-1, length(test_lid_to_owner))
  current = 1
  for i = 1:length(test_lid_to_owner)
    if (test_lid_to_owner[i] == part)
      lid_to_owned_lid[i] = current
      current += 1
    end
  end

  trial_lid_to_gid_extended = copy(trial_lid_to_gid)
  trial_gid_to_lid_extended = copy(trial_gid_to_lid)

  I,J,V = dIJV.part
  IS    = dIS.parts.part
  remote_entries = exchange_entries_vector.part

  length_GI_GJ_GV = count(a->(test_lid_to_owner[a]==part), I)
  for i=1:length(IS.lid_to_owner)
    if (IS.lid_to_owner[i] != part)
      length_GI_GJ_GV += length(remote_entries[i])÷3
    end
  end

  GI = Vector{Int64}(undef,length_GI_GJ_GV)
  GJ = Vector{Int64}(undef,length_GI_GJ_GV)
  GV = Vector{Float64}(undef,length_GI_GJ_GV)

  # Add local entries
  current = 1
  for i = 1:length(I)
    owner = test_lid_to_owner[I[i]]
    if (owner == part)
      GI[current]=lid_to_owned_lid[I[i]]
      GJ[current]=J[i]
      GV[current]=V[i]
      current=current+1
    end
   end

   data = remote_entries.data
   ptrs = remote_entries.ptrs

   # Add remote entries
   for edge_lid = 1:length(IS.lid_to_gid)
     if (IS.lid_to_owner[edge_lid] != part)

      for i = 1:3:length(remote_entries[edge_lid])


        row=Int64(data[ptrs[edge_lid]+i-1])
        GI[current]=lid_to_owned_lid[test_gid_to_lid[row]]
        col=Int64(data[ptrs[edge_lid]+i])
        if (!(haskey(trial_gid_to_lid_extended,col)))
          trial_gid_to_lid_extended[col]=length(trial_lid_to_gid_extended)+1
          push!(trial_lid_to_gid_extended, col)
        end
        GJ[current] = trial_gid_to_lid_extended[col]

        GV[current] = data[ptrs[edge_lid]+i+1]

        current = current + 1
       end
     end
   end
   (GI,GJ,GV,trial_lid_to_gid_extended)
end

function build_fully_assembled_local_portion(m,GI,GJ,GV,trial_lid_to_gid_extended)
  n_owned_dofs = num_owned_entries(m)
  Alocal = sparse_from_coo(
    Gridap.Algebra.SparseMatrixCSR{0,Float64,Int64},
    GI,
    GJ,
    GV,
    n_owned_dofs,
    length(trial_lid_to_gid_extended))
  for i = 1:length(Alocal.colval)
   Alocal.colval[i] = trial_lid_to_gid_extended[Alocal.colval[i]+1] - 1
  end
  Alocal
end

function build_petsc_matrix_from_local_portion(m,n,Alocal)
  # Build PETSc Matrix
  ngrows = num_gids(m)
  ngcols = num_gids(n)
  n_owned_dofs = num_owned_entries(m)
  p = Ref{GridapDistributedPETScWrappers.C.Mat{Float64}}()
  rowptr = _convert_buf_to_petscint(Alocal.rowptr)
  colval = _convert_buf_to_petscint(Alocal.colval)
  GridapDistributedPETScWrappers.C.chk(GridapDistributedPETScWrappers.C.MatCreateMPIAIJWithArrays(
        get_comm(m).comm,
        GridapDistributedPETScWrappers.C.PetscInt(n_owned_dofs),
        GridapDistributedPETScWrappers.C.PetscInt(n_owned_dofs),
        GridapDistributedPETScWrappers.C.PetscInt(ngrows),
        GridapDistributedPETScWrappers.C.PetscInt(ngcols),
        rowptr,
        colval,
        Alocal.nzval,
        p,))
  # NOTE: the triple (rowptr,colval,Alocal.nzval) is passed to the
  #       constructor of Mat() in order to avoid these Julia arrays
  #       from being garbage collected.
  A=GridapDistributedPETScWrappers.Mat{Float64}(p[],(rowptr,colval,Alocal.nzval))
  GridapDistributedPETScWrappers.set_local_to_global_mapping(A,m.IS,n.IS)
  GridapDistributedPETScWrappers.C.chk(GridapDistributedPETScWrappers.C.MatSetOption(p[],
                           GridapDistributedPETScWrappers.C.MAT_NEW_NONZERO_LOCATIONS,
                           GridapDistributedPETScWrappers.C.PETSC_FALSE))
  A.insertmode = GridapDistributedPETScWrappers.C.ADD_VALUES
  A
end

function assemble_global_matrix(
  strat::DistributedAssemblyStrategy{OwnedCellsAssemblyStrategy{false}},
  ::Type{GridapDistributedPETScWrappers.Mat{Float64}},
  IJV::MPIPETScDistributedData,
  m::MPIPETScDistributedIndexSet,
  n::MPIPETScDistributedIndexSet,
)
  dI = DistributedData(get_comm(m),IJV) do part, IJV
    I,_,_ = IJV
    I
  end

  # 1. Determine communication pattern
  dIS, part_to_lst_snd =
    compute_subdomain_graph_dIS_and_lst_snd(m, dI)

  # 2. Communicate entries
  exchange_entries_vector =
    pack_and_comm_entries(dIS, IJV, m, n, part_to_lst_snd)

  # 3. Combine local and remote entries
  GI, GJ, GV, trial_lid_to_gid_extended =
    combine_local_and_remote_entries(dIS, IJV, m, n, exchange_entries_vector)

  # 4. Build fully assembled local portions
  Alocal =
    build_fully_assembled_local_portion(m,GI,GJ,GV,trial_lid_to_gid_extended)

  # 5. Build global matrix from fully assembled local portions
  build_petsc_matrix_from_local_portion(m,n,Alocal)
end

function assemble_global_vector(
  strat::DistributedAssemblyStrategy{OwnedCellsAssemblyStrategy{false}},
  ::Type{GridapDistributedPETScWrappers.Vec{Float64}},
  db::MPIPETScDistributedData,
  indices::MPIPETScDistributedIndexSet)
  vec = allocate_vector(GridapDistributedPETScWrappers.Vec{Float64},indices)
  GridapDistributedPETScWrappers.setindex0!(vec, db.part, indices.lid_to_gid_petsc .- GridapDistributedPETScWrappers.C.PetscInt(1))
  GridapDistributedPETScWrappers.AssemblyBegin(vec)
  GridapDistributedPETScWrappers.AssemblyEnd(vec)
  vec
end

function assemble_global_vector(
  strat::DistributedAssemblyStrategy{OwnedAndGhostCellsAssemblyStrategy{false}},
  ::Type{GridapDistributedPETScWrappers.Vec{Float64}},
  db::MPIPETScDistributedData,
  indices::MPIPETScDistributedIndexSet)
  vec = allocate_vector(GridapDistributedPETScWrappers.Vec{Float64},indices)

  part = MPI.Comm_rank(get_comm(indices).comm)+1
  owned_pos = (indices.parts.part.lid_to_owner .== part)
  bowned    = db.part[owned_pos]
  l2g_petsc = indices.lid_to_gid_petsc[owned_pos] .- GridapDistributedPETScWrappers.C.PetscInt(1)

  GridapDistributedPETScWrappers.setindex0!(vec, bowned, l2g_petsc)
  GridapDistributedPETScWrappers.AssemblyBegin(vec)
  GridapDistributedPETScWrappers.AssemblyEnd(vec)
  vec
end
