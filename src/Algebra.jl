
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

function local_views(a)
  @abstractmethod
end

function get_parts(x)
  return PArrays.get_part_ids(local_views(x))
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

function local_views(a::AbstractPData)
  a
end

function local_views(a::PRange)
  a.partition
end

function local_views(a::PVector)
  a.values
end

function local_views(a::PVector,rows::PRange)
  PArrays.local_view(a,rows)
end

function local_views(a::PSparseMatrix)
  a.values
end

function local_views(a::PSparseMatrix,rows::PRange,cols::PRange)
  PArrays.local_view(a,rows,cols)
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

"""
"""
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

function local_views(a::DistributedCounterCOO,rows,cols)
  @check rows === a.rows
  @check cols === a.cols
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

function change_axes(a::DistributedAllocationCOO{A,B,<:PRange,<:PRange},
                     axes::Tuple{<:PRange,<:PRange}) where {A,B}
  local_axes=map_parts(axes[1].partition,axes[2].partition) do rows,cols
    (Base.OneTo(num_lids(rows)), Base.OneTo(num_lids(cols)))
  end
  allocs=map_parts(change_axes,a.allocs,local_axes)
  DistributedAllocationCOO(a.par_strategy,allocs,axes[1],axes[2])
end

function local_views(a::DistributedAllocationCOO)
  a.allocs
end

function local_views(a::DistributedAllocationCOO,rows,cols)
  @check rows === a.rows
  @check cols === a.cols
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

function Algebra.create_from_nz(a::PSparseMatrix)
  # For FullyAssembledRows the underlying Exchanger should
  # not have ghost layer making assemble! do nothing (TODO check)
  assemble!(a)
  a
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

  # Adjust local matrix size to linear system's index sets
  b=change_axes(a,(rows,cols))

  # Compress local portions
  values = map_parts(create_from_nz,b.allocs)

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

  # Adjust local matrix size to linear system's index sets
  b=change_axes(a,(rows,cols))

  # Compress the local matrices
  values = map_parts(create_from_nz,b.allocs)

  # Wait the transfer to finish
  if t2 !== nothing
    map_parts(schedule,t2)
    map_parts(wait,t2)
  end

  # Finally build the matrix
  # A matrix exchanger will be created under the hood.
  # Building a exchanger can be costly and not always needed.
  # TODO add a more lazy initzialization of this inside PSparseMatrix
  A = PSparseMatrix(values,rows,cols)

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

function local_views(a::PVectorCounter,rows)
  @check rows === a.rows
  a.counters
end

function Arrays.nz_allocation(a::PVectorCounter{<:FullyAssembledRows})
  dofs = a.rows
  values = map_parts(nz_allocation,a.counters)
  PVectorAllocationTrackOnlyValues(a.par_strategy,values,dofs)
end

struct PVectorAllocationTrackOnlyValues{A,B,C}
  par_strategy::A
  values::B
  rows::C
end

function local_views(a::PVectorAllocationTrackOnlyValues)
  a.values
end

function local_views(a::PVectorAllocationTrackOnlyValues,rows)
  @check rows === a.rows
  a.values
end

function Algebra.create_from_nz(a::PVectorAllocationTrackOnlyValues{<:FullyAssembledRows})
  # Create PRange for the rows of the linear system
  parts = get_part_ids(a.values)
  ngdofs = length(a.rows)
  nodofs = map_parts(num_oids,a.rows.partition)
  first_grdof = map_parts(first_gdof_from_ids,a.rows.partition)

  # This one has no ghost rows
  rows = PRange(parts,ngdofs,nodofs,first_grdof)

  _rhs_callback(a,rows)
end

function Algebra.create_from_nz(a::PVectorAllocationTrackOnlyValues{<:SubAssembledRows})
  # This point MUST NEVER be reached. If reached there is an inconsistency
  # in the parallel code in charge of vector assembly
  @assert false
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

function Algebra.create_from_nz(a::PVector)
  # For FullyAssembledRows the underlying Exchanger should
  # not have ghost layer making assemble! do nothing (TODO check)
  assemble!(a)
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
  rows::C
end

function Algebra.create_from_nz(
  a::DistributedAllocationCOO{<:SubAssembledRows},
  c_fespace::PVectorAllocationTrackOnlyValues{<:SubAssembledRows})

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

struct ArrayAllocationTrackTouchedAndValues{A}
  touched::Vector{Bool}
  values::A
end

Gridap.Algebra.LoopStyle(::Type{<:ArrayAllocationTrackTouchedAndValues}) = Gridap.Algebra.Loop()


function local_views(a::PVectorAllocationTrackTouchedAndValues,rows)
  @check rows === a.rows
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
  dofs = b.rows
  values = map_parts(nz_allocation,b.counters)
  B=PVectorAllocationTrackOnlyValues(b.par_strategy,values,dofs)
  A,B
end

function Arrays.nz_allocation(a::PVectorCounter{<:SubAssembledRows})
  dofs = a.rows
  values = map_parts(nz_allocation,a.counters)
  touched = map_parts(values) do values
     fill!(Vector{Bool}(undef,length(values)),false)
  end
  allocations=map_parts(values,touched) do values,touched
    ArrayAllocationTrackTouchedAndValues(touched,values)
  end
  PVectorAllocationTrackTouchedAndValues(allocations,values,dofs)
end

function local_views(a::PVectorAllocationTrackTouchedAndValues)
  a.allocations
end

function Algebra.create_from_nz(a::PVectorAllocationTrackTouchedAndValues)
   parts = get_part_ids(a.values)
   rdofs = a.rows # dof ids of the test space
   ngrdofs = length(rdofs)
   nordofs = map_parts(num_oids,rdofs.partition)
   first_grdof = map_parts(first_gdof_from_ids,rdofs.partition)
   rneigs_snd = rdofs.exchanger.parts_snd
   rneigs_rcv = rdofs.exchanger.parts_rcv

   # Find the ghost rows
   hrow_to_hrdof=map_parts(local_views(a.allocations),rdofs.partition) do allocation, indices
    lids_touched=findall(allocation.touched)
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
