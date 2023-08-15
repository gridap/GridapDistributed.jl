# This might go to Gridap in the future. We keep it here for the moment.
function change_axes(a::Algebra.ArrayCounter,axes)
  @notimplemented
end

# This might go to Gridap in the future. We keep it here for the moment.
function change_axes(a::Algebra.CounterCOO{T,A}, axes::A) where {T,A}
  b=Algebra.CounterCOO{T}(axes)
  b.nnz = a.nnz
  b
end

# This might go to Gridap in the future. We keep it here for the moment.
function change_axes(a::Algebra.AllocationCOO{T,A}, axes::A) where {T,A}
  counter=change_axes(a.counter,axes)
  Algebra.AllocationCOO(counter,a.I,a.J,a.V)
end

# This type is required because MPIArray from PArrays 
# cannot be instantiated with a NULL communicator
struct MPIVoidVector{T} <: AbstractVector{T}
  comm::MPI.Comm
  function MPIVoidVector(::Type{T}) where {T}
    new{T}(MPI.COMM_NULL)
  end
end

Base.size(a::MPIVoidVector) = (0,)
Base.IndexStyle(::Type{<:MPIVoidVector}) = IndexLinear()
function Base.getindex(a::MPIVoidVector,i::Int)
  error("Indexing of MPIVoidVector not possible.")
end
function Base.setindex!(a::MPIVoidVector,v,i::Int)
  error("Indexing of MPIVoidVector not possible.")
end
function Base.show(io::IO,k::MIME"text/plain",data::MPIVoidVector)
  println(io,"MPIVoidVector")
end

function num_parts(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    nparts = MPI.Comm_size(comm)
  else
    nparts = -1
  end
  nparts
end

function get_part_id(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    id = MPI.Comm_rank(comm)+1
  else
    id = -1
  end
  id
end

function i_am_in(comm::MPI.Comm)
  get_part_id(comm) >=0
end

function i_am_in(comm::MPIArray)
  i_am_in(comm.comm)
end

function i_am_in(comm::MPIVoidVector)
  i_am_in(comm.comm)
end

function change_parts(x::Union{MPIArray,DebugArray,Nothing,MPIVoidVector}, new_parts; default=nothing)
  x_new = map(new_parts) do _p
    if isa(x,MPIArray) || isa(x,DebugArray)
      PartitionedArrays.getany(x)
    else
      default
    end
  end
  return x_new
end

function generate_subparts(parts::MPIArray,new_comm_size)
  root_comm = parts.comm
  root_size = MPI.Comm_size(root_comm)
  rank = MPI.Comm_rank(root_comm)

  @static if isdefined(MPI,:MPI_UNDEFINED)
    mpi_undefined = MPI.MPI_UNDEFINED[]
  else
    mpi_undefined = MPI.API.MPI_UNDEFINED[]
  end
  
  if root_size == new_comm_size
    return parts
  else
    if rank < new_comm_size
      comm = MPI.Comm_split(root_comm,0,0)
      return distribute_with_mpi(LinearIndices((new_comm_size,));comm=comm,duplicate_comm=false)
    else
      comm = MPI.Comm_split(root_comm,mpi_undefined,mpi_undefined)
      return MPIVoidVector(eltype(parts))
    end
  end
end

function generate_subparts(parts::DebugArray,new_comm_size)
  DebugArray(LinearIndices((new_comm_size,)))
end

# local_views

function local_views(a)
  @abstractmethod
end

function get_parts(x)
  return linear_indices(local_views(x))
end

function local_views(a::AbstractVector,rows)
  @notimplemented
end

function local_views(a::AbstractMatrix,rows,cols)
  @notimplemented
end

function consistent_local_views(a,ids,isconsistent)
  @abstractmethod
end

function local_views(a::AbstractArray)
  a
end

function local_views(a::PRange)
  partition(a)
end

function local_views(a::PVector)
  partition(a)
end

function local_views(a::PSparseMatrix)
  partition(a)
end

# This function computes a mapping among the local identifiers of a and b
# for which the corresponding global identifiers are both in a and b. 
# Note that the haskey check is necessary because in the general case 
# there might be gids in b which are not present in a
function find_local_to_local_map(a::AbstractLocalIndices,b::AbstractLocalIndices)
  a_local_to_b_local = fill(Int32(-1),local_length(a))
  b_local_to_global  = local_to_global(b)
  a_global_to_local  = global_to_local(a)
  for blid in 1:local_length(b)
    gid = b_local_to_global[blid]
    if a_global_to_local[gid] != zero(eltype(a_global_to_local))
      alid = a_global_to_local[gid]
      a_local_to_b_local[alid] = blid
    end  
  end
  a_local_to_b_local
end

# This type is required in order to be able to access the local portion 
# of distributed sparse matrices and vectors using local indices from the 
# distributed test and trial spaces
struct LocalView{T,N,A,B} <:AbstractArray{T,N}
  plids_to_value::A
  d_to_lid_to_plid::B
  local_size::NTuple{N,Int}
  function LocalView(
    plids_to_value::AbstractArray{T,N},d_to_lid_to_plid::NTuple{N}) where {T,N}
    A = typeof(plids_to_value)
    B = typeof(d_to_lid_to_plid)
    local_size = map(length,d_to_lid_to_plid)
    new{T,N,A,B}(plids_to_value,d_to_lid_to_plid,local_size)
  end
end

Base.size(a::LocalView) = a.local_size
Base.IndexStyle(::Type{<:LocalView}) = IndexCartesian()
function Base.getindex(a::LocalView{T,N},lids::Vararg{Integer,N}) where {T,N}
  plids = map(_lid_to_plid,lids,a.d_to_lid_to_plid)
  if all(i->i>0,plids)
    a.plids_to_value[plids...]
  else
    zero(T)
  end
end
function Base.setindex!(a::LocalView{T,N},v,lids::Vararg{Integer,N}) where {T,N}
  plids = map(_lid_to_plid,lids,a.d_to_lid_to_plid)
  @check all(i->i>0,plids) "You are trying to set a value that is not stored in the local portion"
  a.plids_to_value[plids...] = v
end

function _lid_to_plid(lid,lid_to_plid)
  plid = lid_to_plid[lid]
  plid
end

function local_views(row_partitioned_vector::PVector,test_dofs_partition::PRange)
  if row_partitioned_vector.index_partition === partition(test_dofs_partition)
    @assert false
  else
    map(partition(row_partitioned_vector),
        partition(test_dofs_partition),
        row_partitioned_vector.index_partition) do vector_partition,dofs_partition,row_partition
      LocalView(vector_partition,(find_local_to_local_map(dofs_partition,row_partition),))
    end
  end
end

function local_views(row_col_partitioned_matrix::PSparseMatrix,
                     test_dofs_partition::PRange,
                     trial_dofs_partition::PRange)
    if (row_col_partitioned_matrix.row_partition === partition(test_dofs_partition) || 
      row_col_partitioned_matrix.col_partition === partition(trial_dofs_partition) )
      @assert false                 
    else 
      map(
        partition(row_col_partitioned_matrix),
        partition(test_dofs_partition),
        partition(trial_dofs_partition),
        row_col_partitioned_matrix.row_partition,
        row_col_partitioned_matrix.col_partition) do matrix_partition,
                                                                test_dof_partition,
                                                                trial_dof_partition,
                                                                row_partition,
                                                                col_partition
        rl2lmap = find_local_to_local_map(test_dof_partition,row_partition)
        cl2lmap = find_local_to_local_map(trial_dof_partition,col_partition)
        LocalView(matrix_partition,(rl2lmap,cl2lmap))
      end
    end
end

function change_ghost(a::PVector,ids_fespace::PRange)
  if a.index_partition === partition(ids_fespace)
    a_fespace = a
  else
    a_fespace = similar(a,eltype(a),(ids_fespace,))
    a_fespace .= a
  end
  a_fespace
end

function consistent_local_views(a::PVector,
                                ids_fespace::PRange,
                                isconsistent)
  a_fespace = change_ghost(a,ids_fespace)
  if ! isconsistent
    fetch_vector_ghost_values!(partition(a_fespace),
                               map(reverse,a_fespace.cache)) |> wait
  end
  partition(a_fespace)
end

function Algebra.allocate_vector(::Type{<:PVector{V,A}},ids::PRange) where {V,A}
  values = map(partition(ids)) do ids
    Tv = eltype(A)
    Tv(undef,length(local_to_owner(ids)))
  end
  PVector(values,partition(ids))
end

struct FullyAssembledRows end
struct SubAssembledRows end

# For the moment we use COO format even though
# it is quite memory consuming.
# We need to implement communication in other formats in
# PartitionedArrays.jl

struct PSparseMatrixBuilderCOO{T,B}
  local_matrix_type::Type{T}
  par_strategy::B
end

function Algebra.nz_counter(
  builder::PSparseMatrixBuilderCOO{A}, axs::Tuple{<:PRange,<:PRange}) where A
  test_dofs_gids_prange, trial_dofs_gids_prange = axs
  counters = map(partition(test_dofs_gids_prange),partition(trial_dofs_gids_prange)) do r,c
    axs = (Base.OneTo(local_length(r)),Base.OneTo(local_length(c)))
    Algebra.CounterCOO{A}(axs)
  end
  DistributedCounterCOO(builder.par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
end

"""
"""
struct DistributedCounterCOO{A,B,C,D} <: GridapType
  par_strategy::A
  counters::B
  test_dofs_gids_prange::C
  trial_dofs_gids_prange::D
  function DistributedCounterCOO(
    par_strategy,
    counters::AbstractArray{<:Algebra.CounterCOO},
    test_dofs_gids_prange::PRange,
    trial_dofs_gids_prange::PRange)
    A = typeof(par_strategy)
    B = typeof(counters)
    C = typeof(test_dofs_gids_prange)
    D = typeof(trial_dofs_gids_prange)
    new{A,B,C,D}(par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
  end
end

function local_views(a::DistributedCounterCOO)
  a.counters
end

function local_views(a::DistributedCounterCOO,test_dofs_gids_prange,trial_dofs_gids_prange)
  @check test_dofs_gids_prange === a.test_dofs_gids_prange
  @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
  a.counters
end

function Algebra.nz_allocation(a::DistributedCounterCOO)
  allocs = map(nz_allocation,a.counters)
  DistributedAllocationCOO(a.par_strategy,allocs,a.test_dofs_gids_prange,a.trial_dofs_gids_prange)
end

struct DistributedAllocationCOO{A,B,C,D} <:GridapType
  par_strategy::A
  allocs::B
  test_dofs_gids_prange::C
  trial_dofs_gids_prange::D
  function DistributedAllocationCOO(
    par_strategy,
    allocs::AbstractArray{<:Algebra.AllocationCOO},
    test_dofs_gids_prange::PRange,
    trial_dofs_gids_prange::PRange)
    A = typeof(par_strategy)
    B = typeof(allocs)
    C = typeof(test_dofs_gids_prange)
    D = typeof(trial_dofs_gids_prange)
    new{A,B,C,D}(par_strategy,allocs,test_dofs_gids_prange,trial_dofs_gids_prange)
  end
end

function change_axes(a::DistributedAllocationCOO{A,B,<:PRange,<:PRange},
                     axes::Tuple{<:PRange,<:PRange}) where {A,B}
  local_axes=map(partition(axes[1]),partition(axes[2])) do rows,cols
    (Base.OneTo(local_length(rows)), Base.OneTo(local_length(cols)))
  end
  allocs=map(change_axes,a.allocs,local_axes)
  DistributedAllocationCOO(a.par_strategy,allocs,axes[1],axes[2])
end

function local_views(a::DistributedAllocationCOO)
  a.allocs
end

function local_views(a::DistributedAllocationCOO,test_dofs_gids_prange,trial_dofs_gids_prange)
  @check test_dofs_gids_prange === a.test_dofs_gids_prange
  @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
  a.allocs
end

function first_gdof_from_ids(ids)
  lid_to_gid=local_to_global(ids) 
  owner_to_lid=own_to_local(ids)
  own_length(ids)>0 ? Int(lid_to_gid[first(owner_to_lid)]) : 1
end

function find_gid_and_owner(ighost_to_jghost,jindices)
  jghost_to_local=ghost_to_local(jindices)
  jlocal_to_global=local_to_global(jindices)
  jlocal_to_owner=local_to_owner(jindices)
  ighost_to_jlocal = view(jghost_to_local,ighost_to_jghost)

  ighost_to_global = jlocal_to_global[ighost_to_jlocal]
  ighost_to_owner = jlocal_to_owner[ighost_to_jlocal]
  ighost_to_global, ighost_to_owner
end

function Algebra.create_from_nz(a::PSparseMatrix)
  # For FullyAssembledRows the underlying Exchanger should
  # not have ghost layer making assemble! do nothing (TODO check)
  assemble!(a) |> wait
  a
end

function Algebra.create_from_nz(a::DistributedAllocationCOO{<:FullyAssembledRows})
  f(x) = nothing
  A, = _fa_create_from_nz_with_callback(f,a)
  A
end

# The given ids are assumed to be a sub-set of the lids
function ghost_lids_touched(a::AbstractLocalIndices,gids::AbstractVector{<:Integer})
  i = 0
  ghost_lids_touched = fill(false,ghost_length(a))
  glo_to_loc=global_to_local(a)
  loc_to_gho=local_to_ghost(a)
  for gid in gids
    lid = glo_to_loc[gid]
    ghost_lid = loc_to_gho[lid]
    if ghost_lid > 0 && !ghost_lids_touched[ghost_lid]
      ghost_lids_touched[ghost_lid] = true
      i += 1
    end
  end
  gids_ghost_lid_to_ghost_lid = Vector{Int32}(undef,i)
  i = 0
  ghost_lids_touched .= false
  for gid in gids
    lid = glo_to_loc[gid]
    ghost_lid = loc_to_gho[lid]
    if ghost_lid > 0 && !ghost_lids_touched[ghost_lid]
      ghost_lids_touched[ghost_lid] = true
      i += 1
      gids_ghost_lid_to_ghost_lid[i] = ghost_lid
    end
  end
  gids_ghost_lid_to_ghost_lid
end

# Find the neighbours of partition1 trying 
# to use those in partition2 as a hint 
function _find_neighbours!(partition1, partition2)
  partition2_snd, partition2_rcv = assembly_neighbors(partition2)
  partition2_graph = ExchangeGraph(partition2_snd, partition2_rcv)
  assembly_neighbors(partition1; neighbors=partition2_graph)
end 

function _fa_create_from_nz_with_callback(callback,a)

  # Recover some data
  I,J,V = map(a.allocs) do alloc
    alloc.I, alloc.J, alloc.V
  end |> tuple_of_arrays
  test_dofs_gids_prange = a.test_dofs_gids_prange
  trial_dofs_gids_prange = a.trial_dofs_gids_prange
  test_dofs_gids_partition = partition(test_dofs_gids_prange)
  trial_dofs_gids_partition = partition(trial_dofs_gids_prange)
  ngcdofs = length(trial_dofs_gids_prange)
  nocdofs = map(own_length,trial_dofs_gids_partition)

  rows = _setup_prange_rows_without_ghosts(test_dofs_gids_prange)

  b = callback(rows)

  # convert I and J to global dof ids
  map(to_global!,I,test_dofs_gids_partition)
  map(to_global!,J,trial_dofs_gids_partition)

  # Create the range for cols
  # Note that we are calling here the _setup_rows_prange(...) function even though we 
  # are setting up the range for the cols. Inherent to the FullyAssembledRows() 
  # assembly strategy with a single layer of ghost cells 
  # is the fact that "The global column identifiers of matrix entries 
  # located in rows that a given processor owns have to be such they belong to the set of 
  # ghost DoFs in the local partition of the FE space corresponding to such processor."
  # Any entry in the global matrix sparsity pattern that fulfills this condition is just 
  # ignored in the FullyAssembledRows() assembly strategy with a single layer of ghost cells 
  cols = _setup_rows_prange(trial_dofs_gids_prange,J)


  # Convert again I,J to local numeration
  map(to_local!,I,partition(rows))
  map(to_local!,J,partition(cols))

  # Adjust local matrix size to linear system's index sets
  asys=change_axes(a,(rows,cols))

  # Compress local portions
  values = map(create_from_nz,asys.allocs)

  # Finally build the matrix
  A = PSparseMatrix(values,partition(rows),partition(cols))

  A, b
end

function Algebra.create_from_nz(a::DistributedAllocationCOO{<:SubAssembledRows})
  f(x) = nothing
  A, = _sa_create_from_nz_with_callback(f,f,a)
  A
end

function _generate_column_owner(J,trial_dofs_partition)
  map(J,trial_dofs_partition) do J, indices 
    glo_to_loc=global_to_local(indices) 
    loc_to_own=local_to_owner(indices)
     map(x->loc_to_own[glo_to_loc[x]], J)
  end 
end 

function assemble_coo_with_column_owner!(I,J,Jown,V,row_partition)
  """
    Returns three JaggedArrays with the coo triplets
    to be sent to the corresponding owner parts in parts_snd
  """
  function setup_snd(part,parts_snd,row_lids,coo_entries_with_column_owner)
      global_to_local_row = global_to_local(row_lids)
      local_row_to_owner = local_to_owner(row_lids)
      owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
      ptrs = zeros(Int32,length(parts_snd)+1)
      k_gi, k_gj, k_jo, k_v = coo_entries_with_column_owner
      for k in 1:length(k_gi)
          gi = k_gi[k]
          li = global_to_local_row[gi]
          owner = local_row_to_owner[li]
          if owner != part
              ptrs[owner_to_i[owner]+1] +=1
          end
      end
      PArrays.length_to_ptrs!(ptrs)
      gi_snd_data = zeros(eltype(k_gi),ptrs[end]-1)
      gj_snd_data = zeros(eltype(k_gj),ptrs[end]-1)
      jo_snd_data = zeros(eltype(k_jo),ptrs[end]-1)
      v_snd_data = zeros(eltype(k_v),ptrs[end]-1)
      for k in 1:length(k_gi)
          gi = k_gi[k]
          li = global_to_local_row[gi]
          owner = local_row_to_owner[li]
          if owner != part
              gj = k_gj[k]
              v = k_v[k]
              p = ptrs[owner_to_i[owner]]
              gi_snd_data[p] = gi
              gj_snd_data[p] = gj
              jo_snd_data[p] = k_jo[k]
              v_snd_data[p] = v
              k_v[k] = zero(v)
              ptrs[owner_to_i[owner]] += 1
          end
      end
      PArrays.rewind_ptrs!(ptrs)
      gi_snd = JaggedArray(gi_snd_data,ptrs)
      gj_snd = JaggedArray(gj_snd_data,ptrs)
      jo_snd = JaggedArray(jo_snd_data,ptrs)
      v_snd = JaggedArray(v_snd_data,ptrs)
      gi_snd, gj_snd, jo_snd, v_snd
  end
  """
    Pushes to coo_entries_with_column_owner the tuples 
    gi_rcv,gj_rcv,jo_rcv,v_rcv received from remote processes
  """
  function setup_rcv!(coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
      k_gi, k_gj, k_jo, k_v = coo_entries_with_column_owner
      current_n = length(k_gi)
      new_n = current_n + length(gi_rcv.data)
      resize!(k_gi,new_n)
      resize!(k_gj,new_n)
      resize!(k_jo,new_n)
      resize!(k_v,new_n)
      for p in 1:length(gi_rcv.data)
          k_gi[current_n+p] = gi_rcv.data[p]
          k_gj[current_n+p] = gj_rcv.data[p]
          k_jo[current_n+p] = jo_rcv.data[p]
          k_v[current_n+p] = v_rcv.data[p]
      end
  end
  part = linear_indices(row_partition)
  parts_snd, parts_rcv = assembly_neighbors(row_partition)
  coo_entries_with_column_owner = map(tuple,I,J,Jown,V)
  gi_snd, gj_snd, jo_snd, v_snd = map(setup_snd,part,parts_snd,row_partition,coo_entries_with_column_owner) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t1 = exchange(gi_snd,graph)
  t2 = exchange(gj_snd,graph)
  t3 = exchange(jo_snd,graph)
  t4 = exchange(v_snd,graph)
  @async begin
      gi_rcv = fetch(t1)
      gj_rcv = fetch(t2)
      jo_rcv = fetch(t3)
      v_rcv = fetch(t4)
      map(setup_rcv!,coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
      I,J,Jown,V
  end
end


function _sa_create_from_nz_with_callback(callback,async_callback,a)
  # Recover some data
  I,J,V = map(a.allocs) do alloc
    alloc.I, alloc.J, alloc.V
  end |> tuple_of_arrays
  test_dofs_gids_prange = a.test_dofs_gids_prange
  trial_dofs_gids_prange = a.trial_dofs_gids_prange
  test_dofs_gids_partition = partition(test_dofs_gids_prange)
  trial_dofs_gids_partition = partition(trial_dofs_gids_prange)
  ngrdofs = length(test_dofs_gids_prange)
  ngcdofs = length(test_dofs_gids_prange)

  # convert I and J to global dof ids
  map(to_global!,I,test_dofs_gids_partition)
  map(to_global!,J,trial_dofs_gids_partition)

  # Create the Prange for the rows
  rows = _setup_rows_prange(test_dofs_gids_prange,I)
  
  # Move (I,J,V) triplets to the owner process of each row I.
  # The triplets are accompanyed which Jo which is the process column owner
  Jo=_generate_column_owner(J,trial_dofs_gids_partition)
  t=assemble_coo_with_column_owner!(I,J,Jo,V,partition(rows))

  # Here we can overlap computations
  # This is a good place to overlap since
  # sending the matrix rows is a lot of data
  b = callback(rows)

  # Wait the transfer to finish
  wait(t)

  # Create the Prange for the cols
  cols = _setup_cols_prange(trial_dofs_gids_prange,J,Jo)

  # Overlap rhs communications with CSC compression
  t2 = async_callback(b)

  # Convert again I,J to local numeration
  map(to_local!,I,partition(rows))
  map(to_local!,J,partition(cols))

  # Adjust local matrix size to linear system's index sets
  asys=change_axes(a,(rows,cols))

  # Compress the local matrices
  values = map(create_from_nz,asys.allocs)

  # Wait the transfer to finish
  if t2 !== nothing
    wait(t2)
  end

  # Finally build the matrix
  A = PSparseMatrix(values,partition(rows),partition(cols))

  A, b
end

struct PVectorBuilder{T,B}
  local_vector_type::Type{T}
  par_strategy::B
end

function Algebra.nz_counter(builder::PVectorBuilder,axs::Tuple{<:PRange})
  T = builder.local_vector_type
  rows, = axs
  counters = map(partition(rows)) do rows
    axs = (Base.OneTo(local_length(rows)),)
    nz_counter(ArrayBuilder(T),axs)
  end
  PVectorCounter(builder.par_strategy,counters,rows)
end

struct PVectorCounter{A,B,C}
  par_strategy::A
  counters::B
  test_dofs_gids_prange::C
end

Algebra.LoopStyle(::Type{<:PVectorCounter}) = DoNotLoop()

function local_views(a::PVectorCounter)
  a.counters
end

function local_views(a::PVectorCounter,rows)
  @check rows === a.test_dofs_gids_prange
  a.counters
end

function Arrays.nz_allocation(a::PVectorCounter{<:FullyAssembledRows})
  dofs = a.test_dofs_gids_prange
  values = map(nz_allocation,a.counters)
  PVectorAllocationTrackOnlyValues(a.par_strategy,values,dofs)
end

struct PVectorAllocationTrackOnlyValues{A,B,C}
  par_strategy::A
  values::B
  test_dofs_gids_prange::C
end

function local_views(a::PVectorAllocationTrackOnlyValues)
  a.values
end

function local_views(a::PVectorAllocationTrackOnlyValues,rows)
  @check rows === a.test_dofs_gids_prange
  a.values
end

# Create PRange for the rows of the linear system
# without local ghost dofs as per required in the 
# FullyAssembledRows() parallel assembly strategy 
function _setup_prange_rows_without_ghosts(test_dofs_gids_prange)
  ngdofs = length(test_dofs_gids_prange)
  test_dofs_gids_partition = partition(test_dofs_gids_prange)
  nodofs = map(own_length,test_dofs_gids_partition)
  rindices=map(test_dofs_gids_partition) do dofs_indices 
    owner = part_id(dofs_indices)
    own_indices=OwnIndices(ngdofs,owner,own_to_global(dofs_indices))
    ghost_indices=GhostIndices(ngdofs,Int64[],Int32[])
    OwnAndGhostIndices(own_indices,ghost_indices)
  end
  PRange(rindices)
end

function _setup_rows_prange(test_dofs_gids_prange,I)
  ngdofs = length(test_dofs_gids_prange)
  test_dofs_gids_partition = partition(test_dofs_gids_prange)
  I_ghost_lids_to_test_dofs_ghost_lids = map(ghost_lids_touched,test_dofs_gids_partition,I)
  _setup_rows_prange_impl_(ngdofs,I_ghost_lids_to_test_dofs_ghost_lids,test_dofs_gids_partition)
end

function _setup_rows_prange_impl_(ngdofs,I_ghost_lids_to_dofs_ghost_lids,test_dofs_gids_partition)
  gids_ghost_to_global, gids_ghost_to_owner = map(
    find_gid_and_owner,I_ghost_lids_to_dofs_ghost_lids,test_dofs_gids_partition) |> tuple_of_arrays

  indices=map(test_dofs_gids_partition, 
               gids_ghost_to_global, 
               gids_ghost_to_owner) do dofs_indices, ghost_to_global, ghost_to_owner 
     owner = part_id(dofs_indices)
     own_indices=OwnIndices(ngdofs,owner,own_to_global(dofs_indices))
     ghost_indices=GhostIndices(ngdofs,ghost_to_global,ghost_to_owner)
     OwnAndGhostIndices(own_indices,ghost_indices)
  end
  # Here we are assuming that the sparse communication graph underlying test_dofs_gids_partition
  # is a superset of the one underlying indices. This is (has to be) true for the rows of the linear system.
  # The precondition required for the consistency of any parallel assembly process in GridapDistributed 
  # is that each processor can determine locally with a single layer of ghost cells the global indices and associated 
  # processor owners of the rows that it touches after assembly of integration terms posed on locally-owned entities 
  # (i.e., either cells or faces). 
  _find_neighbours!(indices, test_dofs_gids_partition)
  PRange(indices)
end 

function _setup_cols_prange(trial_dofs_gids_prange,J,Jown)
  ngdofs = length(trial_dofs_gids_prange)
  trial_dofs_gids_partition = partition(trial_dofs_gids_prange)
   
  J_ghost_to_global, J_ghost_to_owner = map(J,Jown,trial_dofs_gids_partition) do J, Jown, trial_dofs_gids_partition
    ghost_touched=Dict{Int,Bool}()
    ghost_to_global=Int64[] 
    ghost_to_owner=Int64[]
    me=part_id(trial_dofs_gids_partition)
    for (j,jo) in zip(J,Jown)
      if jo!=me
        if !haskey(ghost_touched,j)
          push!(ghost_to_global,j)
          push!(ghost_to_owner,jo)
          ghost_touched[j]=true
        end
      end
    end
    ghost_to_global, ghost_to_owner
  end |> tuple_of_arrays

  indices=map(trial_dofs_gids_partition, 
              J_ghost_to_global,
              J_ghost_to_owner) do dofs_indices, ghost_to_global, ghost_to_owner 
     owner = part_id(dofs_indices)
     own_indices=OwnIndices(ngdofs,owner,own_to_global(dofs_indices))
     ghost_indices=GhostIndices(ngdofs,ghost_to_global,ghost_to_owner)
     OwnAndGhostIndices(own_indices,ghost_indices)
  end
  # Here we cannot assume that the sparse communication graph underlying 
  # trial_dofs_gids_partition is a superset of the one underlying indices.
  # Here we chould check whether it is included and call _find_neighbours!()
  # if this is the case. At present, we are not taking advantage of this, 
  # but let the parallel scalable algorithm to compute the reciprocal to do the 
  # job. 
  PRange(indices)
end 

function Algebra.create_from_nz(a::PVectorAllocationTrackOnlyValues{<:FullyAssembledRows})
  rows = _setup_prange_rows_without_ghosts(a.test_dofs_gids_prange)
  _rhs_callback(a,rows)
end

function Algebra.create_from_nz(a::PVectorAllocationTrackOnlyValues{<:SubAssembledRows})
  # This point MUST NEVER be reached. If reached there is an inconsistency
  # in the parallel code in charge of vector assembly
  @assert false
end

function _rhs_callback(row_partitioned_vector_partition,rows)
  # The ghost values in row_partitioned_vector_partition are 
  # aligned with the FESpace but not with the ghost values in the rows of A
  b_fespace = PVector(row_partitioned_vector_partition.values,
                      partition(row_partitioned_vector_partition.test_dofs_gids_prange))

  # This one is aligned with the rows of A
  b = similar(b_fespace,eltype(b_fespace),(rows,))

  # First transfer owned values
  b .= b_fespace

  # Now transfer ghost
  function transfer_ghost(b,b_fespace,ids,ids_fespace)
    num_ghosts_vec = ghost_length(ids)
    gho_to_loc_vec = ghost_to_local(ids)
    loc_to_glo_vec = local_to_global(ids)
    gid_to_lid_fe  = global_to_local(ids_fespace)
    for ghost_lid_vec in 1:num_ghosts_vec
      lid_vec     = gho_to_loc_vec[ghost_lid_vec]
      gid         = loc_to_glo_vec[lid_vec]
      lid_fespace = gid_to_lid_fe[gid]
      b[lid_vec] = b_fespace[lid_fespace]
    end
  end
  map(
    transfer_ghost,
    partition(b),
    partition(b_fespace),
    b.index_partition,
    b_fespace.index_partition)

  return b
end

function Algebra.create_from_nz(a::PVector)
  assemble!(a) |> wait
  a
end

function Algebra.create_from_nz(
  a::DistributedAllocationCOO{<:FullyAssembledRows},
  c_fespace::PVectorAllocationTrackOnlyValues{<:FullyAssembledRows})

  function callback(rows)
    _rhs_callback(c_fespace,rows)
  end

  A,b = _fa_create_from_nz_with_callback(callback,a)
  A,b
end

struct PVectorAllocationTrackTouchedAndValues{A,B,C}
  allocations::A
  values::B
  test_dofs_gids_prange::C
end

function Algebra.create_from_nz(
  a::DistributedAllocationCOO{<:SubAssembledRows},
  c_fespace::PVectorAllocationTrackOnlyValues{<:SubAssembledRows})

  function callback(rows)
    _rhs_callback(c_fespace,rows)
  end

  function async_callback(b)
    # now we can assemble contributions
    assemble!(b)
  end

  A,b = _sa_create_from_nz_with_callback(callback,async_callback,a)
  A,b
end

struct ArrayAllocationTrackTouchedAndValues{A}
  touched::Vector{Bool}
  values::A
end

Gridap.Algebra.LoopStyle(::Type{<:ArrayAllocationTrackTouchedAndValues}) = Gridap.Algebra.Loop()


function local_views(a::PVectorAllocationTrackTouchedAndValues,rows)
  @check rows === a.test_dofs_gids_prange
  a.allocations
end

@inline function Arrays.add_entry!(c::Function,a::ArrayAllocationTrackTouchedAndValues,v,i,j)
  @notimplemented
end
@inline function Arrays.add_entry!(c::Function,a::ArrayAllocationTrackTouchedAndValues,v,i)
  if i>0
    if !(a.touched[i])
      a.touched[i]=true
    end
    if v!=nothing
      a.values[i]=c(v,a.values[i])
    end
  end
  nothing
end
@inline function Arrays.add_entries!(c::Function,a::ArrayAllocationTrackTouchedAndValues,v,i,j)
  @notimplemented
end
@inline function Arrays.add_entries!(c::Function,a::ArrayAllocationTrackTouchedAndValues,v,i)
  if v != nothing
    for (ve,ie) in zip(v,i)
      Arrays.add_entry!(c,a,ve,ie)
    end
  else
    for ie in i
      Arrays.add_entry!(c,a,nothing,ie)
    end
  end
  nothing
end

function Arrays.nz_allocation(a::DistributedCounterCOO{<:SubAssembledRows},
                              b::PVectorCounter{<:SubAssembledRows})
  A = nz_allocation(a)
  dofs = b.test_dofs_gids_prange
  values = map(nz_allocation,b.counters)
  B=PVectorAllocationTrackOnlyValues(b.par_strategy,values,dofs)
  A,B
end

function Arrays.nz_allocation(a::PVectorCounter{<:SubAssembledRows})
  dofs = a.test_dofs_gids_prange
  values = map(nz_allocation,a.counters)
  touched = map(values) do values
     fill!(Vector{Bool}(undef,length(values)),false)
  end
  allocations=map(values,touched) do values,touched
    ArrayAllocationTrackTouchedAndValues(touched,values)
  end
  PVectorAllocationTrackTouchedAndValues(allocations,values,dofs)
end

function local_views(a::PVectorAllocationTrackTouchedAndValues)
  a.allocations
end

function Algebra.create_from_nz(a::PVectorAllocationTrackTouchedAndValues)
  test_dofs_prange = a.test_dofs_gids_prange # dof ids of the test space
  test_dofs_prange_partition = partition(test_dofs_prange)
  ngrdofs = length(test_dofs_prange)
   
  # Find the ghost rows
  I_ghost_lids_to_dofs_ghost_lids=map(local_views(a.allocations),test_dofs_prange_partition) do allocation, indices
    dofs_lids_touched=findall(allocation.touched)
    loc_to_gho = local_to_ghost(indices)
    n_I_ghost_lids = count((x)->loc_to_gho[x]!=0,dofs_lids_touched)
    I_ghost_lids = Vector{Int32}(undef,n_I_ghost_lids)
    cur=1
    for lid in dofs_lids_touched
      dof_lid=loc_to_gho[lid]
      if dof_lid!=0
        I_ghost_lids[cur]=dof_lid
        cur=cur+1
      end
    end
    I_ghost_lids
  end

  rows = _setup_rows_prange_impl_(ngrdofs,
                             I_ghost_lids_to_dofs_ghost_lids,
                             test_dofs_prange_partition)

  b = _rhs_callback(a,rows)
  t2 = assemble!(b)

   # Wait the transfer to finish
   if t2 !== nothing
     wait(t2)
   end
   b
end
