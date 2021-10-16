
function local_views(a::AbstractVector)
  @abstractmethod
end

function consistent_local_views(a,ids,isconsistent)
  @abstractmethod
end

function local_views(a::AbstractPData)
  a
end

function local_views(a::PVector)
  a.values
end

function local_views(a::PRange)
  a.partition
end

function change_ghost(a::PVector,ids_fespace::PRange)
  if a.rows === ids_fespace
    a_fespace = a
  else
    a_fespace = similar(a,eltype(a),ids_fespace)
    a_fespace .= a
  end
  a_fespace
end

function consistent_local_views(a::PVector,ids_fespace::PRange,isconsistent)
  a_fespace = change_ghost(a,ids_fespace)
  if ! isconsistent
    exchange!(a_fespace)
  end
  a_fespace.values
end

function Algebra.allocate_vector(::Type{<:PVector{T,A}},ids::PRange) where {T,A}
  values = map_parts(ids.partition) do ids
    Tv = eltype(A)
    Tv(undef,num_lids(ids))
  end
  PVector(values,ids)
end

function local_views(a::PSparseMatrix)
  a.values
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
  rows, cols = axs
  counters = map_parts(rows.partition,cols.partition) do r,c
    axs = (Base.OneTo(num_lids(r)),Base.OneTo(num_lids(c)))
    Algebra.CounterCOO{A}(axs)
  end
  DistributedCounterCOO(builder.par_strategy,counters,rows,cols)
end

struct DistributedCounterCOO{A,B,C,D} <: GridapType
  par_strategy::A
  counters::B
  rows::C
  cols::D
  function DistributedCounterCOO(
    par_strategy,
    counters::AbstractPData{<:Algebra.CounterCOO},
    rows::PRange,
    cols::PRange)
    A = typeof(par_strategy)
    B = typeof(counters)
    C = typeof(rows)
    D = typeof(cols)
    new{A,B,C,D}(par_strategy,counters,rows,cols)
  end
end

function local_views(a::DistributedCounterCOO)
  a.counters
end

function Algebra.nz_allocation(a::DistributedCounterCOO)
  allocs = map_parts(nz_allocation,a.counters)
  DistributedAllocationCOO(a.par_strategy,allocs,a.rows,a.cols)
end

struct DistributedAllocationCOO{A,B,C,D} <:GridapType
  par_strategy::A
  allocs::B
  rows::C
  cols::D
  function DistributedAllocationCOO(
    par_strategy,
    allocs::AbstractPData{<:Algebra.AllocationCOO},
    rows::PRange,
    cols::PRange)
    A = typeof(par_strategy)
    B = typeof(allocs)
    C = typeof(rows)
    D = typeof(cols)
    new{A,B,C,D}(par_strategy,allocs,rows,cols)
  end
end

function local_views(a::DistributedAllocationCOO)
  a.allocs
end

function first_gdof_from_ids(ids)
  num_oids(ids)>0 ? Int(ids.lid_to_gid[ids.oid_to_lid[1]]) : 1
end

function find_gid_and_part(hid_to_hdof,dofs)
  hid_to_ldof = view(dofs.hid_to_lid,hid_to_hdof)
  hid_to_gid = dofs.lid_to_gid[hid_to_ldof]
  hid_to_part = dofs.lid_to_part[hid_to_ldof]
  hid_to_gid, hid_to_part
end

function Algebra.create_from_nz(a::DistributedAllocationCOO{<:FullyAssembledRows})
  f(x) = nothing
  A, = _fa_create_from_nz_with_callback(f,a)
  A
end

function _fa_create_from_nz_with_callback(callback,a)

  # Recover some data
  I,J,C = map_parts(a.allocs) do alloc
    alloc.I, alloc.J, alloc.V
  end
  parts = get_part_ids(a.allocs)
  rdofs = a.rows # dof ids of the test space
  cdofs = a.cols # dof ids of the trial space
  ngrdofs = length(rdofs)
  ngcdofs = length(cdofs)
  nordofs = map_parts(num_oids,rdofs.partition)
  nocdofs = map_parts(num_oids,cdofs.partition)
  first_grdof = map_parts(first_gdof_from_ids,rdofs.partition)
  first_gcdof = map_parts(first_gdof_from_ids,cdofs.partition)
  cneigs_snd = cdofs.exchanger.parts_snd
  cneigs_rcv = cdofs.exchanger.parts_rcv

  # This one has not ghost rows
  rows = PRange(
    parts,
    ngrdofs,
    nordofs,
    first_grdof)

  callback_output = callback(rows)

  # convert I and J to global dof ids
  to_gids!(I,rdofs)
  to_gids!(J,cdofs)

  # Find the ghost cols
  hcol_to_hcdof = touched_hids(cdofs,J)
  hcol_to_gid, hcol_to_part = map_parts(
    find_gid_and_part,hcol_to_hcdof,cdofs.partition)

  # Create the range for cols
  cols = PRange(
    parts,
    ngcdofs,
    nocdofs,
    first_gcdof,
    hcol_to_gid,
    hcol_to_part,
    cneigs_snd,
    cneigs_rcv)

  # Convert again I,J to local numeration
  to_lids!(I,rows)
  to_lids!(J,cols)

  # Compress local portions
  values = map_parts(create_from_nz,a.allocs)

  # Build the matrix exchanger. This can be empty since no ghost rows
  exchanger = empty_exchanger(parts)

  # Finally build the matrix
  A = PSparseMatrix(values,rows,cols,exchanger)

  A, callback_output
end

function Algebra.create_from_nz(a::DistributedAllocationCOO{<:SubAssembledRows})
  f(x) = nothing
  A, = _sa_create_from_nz_with_callback(f,f,a)
  A
end

function _sa_create_from_nz_with_callback(callback,async_callback,a)

  # Recover some data
  I,J,C = map_parts(a.allocs) do alloc
    alloc.I, alloc.J, alloc.V
  end
  parts = get_part_ids(a.allocs)
  rdofs = a.rows # dof ids of the test space
  cdofs = a.cols # dof ids of the trial space
  ngrdofs = length(rdofs)
  ngcdofs = length(cdofs)
  nordofs = map_parts(num_oids,rdofs.partition)
  nocdofs = map_parts(num_oids,cdofs.partition)
  first_grdof = map_parts(first_gdof_from_ids,rdofs.partition)
  first_gcdof = map_parts(first_gdof_from_ids,cdofs.partition)
  rneigs_snd = rdofs.exchanger.parts_snd
  rneigs_rcv = rdofs.exchanger.parts_rcv
  cneigs_snd = cdofs.exchanger.parts_snd
  cneigs_rcv = cdofs.exchanger.parts_rcv

  # convert I and J to global dof ids
  to_gids!(I,rdofs)
  to_gids!(J,cdofs)

  # Find the ghost rows
  hrow_to_hrdof = touched_hids(rdofs,I)
  hrow_to_gid, hrow_to_part = map_parts(
    find_gid_and_part,hrow_to_hrdof,rdofs.partition)

  # Create the range for rows
  rows = PRange(
    parts,
    ngrdofs,
    nordofs,
    first_grdof,
    hrow_to_gid,
    hrow_to_part,
    rneigs_snd,
    rneigs_rcv)

  # Move values to the owner part
  # since we have integrated only over owned cells
  t = async_assemble!(I,J,C,rows)

  # Here we can overlap computations
  # This is a good place to overlap since
  # sending the matrix rows is a lot of data
  callback_output = callback(rows)

  # Wait the transfer to finish
  map_parts(schedule,t)
  map_parts(wait,t)

  # Find the ghost cols
  hcol_to_hcdof = touched_hids(cdofs,J)
  hcol_to_gid, hcol_to_part = map_parts(
    find_gid_and_part,hcol_to_hcdof,cdofs.partition)

  # Create the range for cols
  cols = PRange(
    parts,
    ngcdofs,
    nocdofs,
    first_gcdof,
    hcol_to_gid,
    hcol_to_part,
    cneigs_snd,
    cneigs_rcv)

  # Overlap rhs communications with CSC compression
  t2 = async_callback(callback_output)

  # Convert again I,J to local numeration
  to_lids!(I,rows)
  to_lids!(J,cols)

  # Compress the local matrices
  values = map_parts(create_from_nz,a.allocs)

  # Wait the transfer to finish
  if t2 !== nothing
    map_parts(schedule,t2)
    map_parts(wait,t2)
  end

  # Build the matrix exchanger
  # TODO for the moment, we build an empty exchanger
  # (fine for problem that do not need to update the matrix)
  exchanger = empty_exchanger(parts)

  # TODO building a non-empty exchanger
  # can be costly and not always needed
  # add a more lazy initzialization of this inside PSparseMatrix

  # Finally build the matrix
  A = PSparseMatrix(values,rows,cols,exchanger)

  A, callback_output
end

struct PVectorBuilder{T,B}
  local_vector_type::Type{T}
  par_strategy::B
end

function Algebra.nz_counter(builder::PVectorBuilder,axs::Tuple{<:PRange})
  T = builder.local_vector_type
  rows, = axs
  counters = map_parts(rows.partition) do rows
    axs = (Base.OneTo(num_lids(rows)),)
    nz_counter(ArrayBuilder(T),axs)
  end
  PVectorCounter(builder.par_strategy,counters,rows)
end

struct PVectorCounter{A,B,C}
  par_strategy::A
  counters::B
  rows::C
end

Algebra.LoopStyle(::Type{<:PVectorCounter}) = DoNotLoop()

function local_views(a::PVectorCounter)
  a.counters
end

function Arrays.nz_allocation(a::PVectorCounter)
  dofs = a.rows
  values = map_parts(nz_allocation,a.counters)
  PVectorAllocation(a.par_strategy,values,dofs)
end

struct PVectorAllocation{A,B,C}
  par_strategy::A
  values::B
  rows::C
end

function local_views(a::PVectorAllocation)
  a.values
end

function Algebra.create_from_nz(a::PVectorAllocation{<:FullyAssembledRows})
  # 1. Create PRange for the rows of the linear system
  parts = get_part_ids(a.values)
  rdofs = a.rows # dof ids of the test space
  ngrdofs = length(rdofs)
  nordofs = map_parts(num_oids,rdofs.partition)
  first_grdof = map_parts(first_gdof_from_ids,rdofs.partition)
  # This one has not ghost rows
  rows = PRange(
    parts,
    ngrdofs,
    nordofs,
    first_grdof)
  # 2. Transform data to output vector without communication
  _rhs_callback(a,rows)
end

function Algebra.create_from_nz(a::PVectorAllocation{<:SubAssembledRows})
  # PVectorAllocation{<:SubAssembledRows} does not provide the information
  # required to be able to build a PVector out of it. A different
  # ArrayBuilder/ArrayCounter/ArrayAllocation set is needed
  @notimplemented
end

function _rhs_callback(c_fespace,rows)

  # The ghost values in b_fespace are aligned with the FESpace
  # but not with the ghost values in the rows of A
  b_fespace = PVector(c_fespace.values,c_fespace.rows)

  # This one is aligned with the rows of A
  b = similar(b_fespace,eltype(b_fespace),(rows,))

  # First transfer owned values
  b .= b_fespace

  # Now transfer ghost
  function transfer_ghost(b,b_fespace,ids,ids_fespace)
    for hid in 1:num_hids(ids)
      lid = ids.hid_to_lid[hid]
      gid = ids.lid_to_gid[lid]
      lid_fespace = ids_fespace.gid_to_lid[gid]
      b[lid] = b_fespace[lid_fespace]
    end
  end
  map_parts(
    transfer_ghost,
    b.values,
    b_fespace.values,
    b.rows.partition,
    b_fespace.rows.partition)

  return b
end

function Algebra.create_from_nz(
  a::DistributedAllocationCOO{<:FullyAssembledRows},
  c_fespace::PVectorAllocation{<:FullyAssembledRows})

  function callback(rows)
    _rhs_callback(c_fespace,rows)
  end

  A,b = _fa_create_from_nz_with_callback(callback,a)
  A,b
end

function Algebra.create_from_nz(
  a::DistributedAllocationCOO{<:SubAssembledRows},
  c_fespace::PVectorAllocation{<:SubAssembledRows})

  function callback(rows)
    _rhs_callback(c_fespace,rows)
  end

  function async_callback(b)
    # now we can assemble contributions
    async_assemble!(b)
  end

  A,b = _sa_create_from_nz_with_callback(callback,async_callback,a)
  A,b
end

################### Experimental (required for assemble_vector + SubAssembledRows)

struct PVectorBuilderSubAssembledRows{T}
  local_vector_type::Type{T}
end

struct ArrayCounterSubAssembledRows{T,A}
  axes::A
  touched::Vector{Bool}
  function ArrayCounterSubAssembledRows{T}(axes::A) where {T,A<:Tuple{Vararg{AbstractUnitRange}}}
    size=map(length,axes)
    new{T,A}(axes,fill!(Array{Bool}(undef,size),false))
  end
end

Gridap.Algebra.LoopStyle(::Type{<:ArrayCounterSubAssembledRows}) = Gridap.Algebra.Loop()
@inline function Arrays.add_entry!(c::Function,a::ArrayCounterSubAssembledRows,v,i,j)
  @notimplemented
end
@inline function Arrays.add_entry!(c::Function,a::ArrayCounterSubAssembledRows,v,i)
  if i>0 && ! (a.touched[i])
    a.touched[i]=true
  end
  nothing
end
@inline function Arrays.add_entries!(c::Function,a::ArrayCounterSubAssembledRows,v,i,j)
  @notimplemented
end
@inline function Arrays.add_entries!(c::Function,a::ArrayCounterSubAssembledRows,v,i)
  for ie in i
    Arrays.add_entry!(c,a,v,ie)
  end
  nothing
end
Arrays.nz_allocation(a::ArrayCounterSubAssembledRows{T}) where T = fill!(similar(T,map(length,a.axes)),zero(eltype(T)))


struct PVectorCounterSubAssembledRows{A,B}
  counters::A
  rows::B
end

function local_views(a::PVectorCounterSubAssembledRows)
  a.counters
end

function Algebra.nz_counter(builder::PVectorBuilderSubAssembledRows,axs::Tuple{<:PRange})
  T = builder.local_vector_type
  rows, = axs
  counters = map_parts(rows.partition) do rows
    axs = (Base.OneTo(num_lids(rows)),)
    ArrayCounterSubAssembledRows{T}(axs)
  end
  PVectorCounterSubAssembledRows(counters,rows)
end

struct PVectorAllocationSubAssembledRows{A,B,C}
  counters::A
  values::B
  rows::C
end

function Arrays.nz_allocation(a::PVectorCounterSubAssembledRows)
  dofs = a.rows
  values = map_parts(nz_allocation,a.counters)
  PVectorAllocationSubAssembledRows(a.counters,values,dofs)
end

function local_views(a::PVectorAllocationSubAssembledRows)
  a.values
end

function Gridap.FESpaces.symbolic_loop_vector!(b,a::GenericSparseMatrixAssembler,vecdata)
  get_vec(a::Tuple) = a[1]
  get_vec(a) = a
  if Gridap.Algebra.LoopStyle(b) == Gridap.Algebra.DoNotLoop()
    return b
  end
  for (cellvec,_cellids) in zip(vecdata...)
    cellids = Gridap.FESpaces.map_cell_rows(a.strategy,_cellids)
    rows_cache = array_cache(cellids)
    if length(cellids) > 0
      vec1 = get_vec(first(cellvec))
      rows1 = getindex!(rows_cache,cellids,1)
      touch! = TouchEntriesMap()
      touch_cache = return_cache(touch!,b,vec1,rows1)
      caches = touch_cache, rows_cache
      _symbolic_loop_vector!(b,caches,cellids,vec1)
    end
  end
  b
end

@noinline function _symbolic_loop_vector!(A,caches,cellids,vec1)
  touch_cache, rows_cache = caches
  touch! = TouchEntriesMap()
  for cell in 1:length(cellids)
    rows = getindex!(rows_cache,cellids,cell)
    evaluate!(touch_cache,touch!,A,vec1,rows)
  end
end

function Algebra.create_from_nz(a::PVectorAllocationSubAssembledRows)
   parts = get_part_ids(a.values)
   rdofs = a.rows # dof ids of the test space
   ngrdofs = length(rdofs)
   nordofs = map_parts(num_oids,rdofs.partition)
   first_grdof = map_parts(first_gdof_from_ids,rdofs.partition)
   rneigs_snd = rdofs.exchanger.parts_snd
   rneigs_rcv = rdofs.exchanger.parts_rcv

   # Find the ghost rows
   hrow_to_hrdof=map_parts(a.counters,rdofs.partition) do counter, indices
    lids_touched=findall(counter.touched)
    nhlids = count((x)->indices.lid_to_ohid[x]<0,lids_touched)
    hlids = Vector{Int32}(undef,nhlids)
    cur=1
    for lid in lids_touched
      hlid=indices.lid_to_ohid[lid]
      if hlid<0
        hlids[cur]=-hlid
        cur=cur+1
      end
    end
    hlids
   end
   hrow_to_gid, hrow_to_part = map_parts(
       find_gid_and_part,hrow_to_hrdof,rdofs.partition)

   # Create the range for rows
   rows = PRange(
           parts,
           ngrdofs,
           nordofs,
           first_grdof,
           hrow_to_gid,
           hrow_to_part,
           rneigs_snd,
           rneigs_rcv)

   b = _rhs_callback(a,rows)
   t2 = async_assemble!(b)

   # Wait the transfer to finish
   if t2 !== nothing
     map_parts(schedule,t2)
     map_parts(wait,t2)
   end
   b
end
