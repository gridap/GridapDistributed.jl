
function local_views(a::AbstractVector)
  @abstractmethod
end

function consistent_local_views(a,ids)
  @abstractmethod
end

function local_views(a::AbstractPData)
  a
end

function local_views(a::PVector)
  a.values
end

function consistent_local_views(a::PVector,ids_fespace::PRange)
  if a.rows === ids_fespace
    a_fespace = a
  else
    a_fespace = similar(a,eltype(a),ids_fespace)
    a_fespace .= a
  end
  exchange!(a_fespace)
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

# For the moment we use COO format even though
# it is quite memory consuming.
# We need to implement communication in other formats in
# PartitionedArrays.jl

struct PSparseMatrixBuilderCOO{T}
  local_matrix_type::Type{T}
end

function Algebra.nz_counter(
  builder::PSparseMatrixBuilderCOO{A}, axs::Tuple{<:PRange,<:PRange}) where A
  rows, cols = axs
  counters = map_parts(rows.partition,cols.partition) do r,c
    axs = (Base.OneTo(num_lids(r)),Base.OneTo(num_lids(c)))
    Algebra.CounterCOO{A}(axs)
  end
  DistributedCounterCOO(counters,rows,cols)
end

struct DistributedCounterCOO{A,B,C} <: GridapType
  counters::A
  rows::B
  cols::C
  function DistributedCounterCOO(
    counters::AbstractPData{<:Algebra.CounterCOO},
    rows::PRange,
    cols::PRange)
    A = typeof(counters)
    B = typeof(rows)
    C = typeof(cols)
    new{A,B,C}(counters,rows,cols)
  end
end

function local_views(a::DistributedCounterCOO)
  a.counters
end

function Algebra.nz_allocation(a::DistributedCounterCOO)
  allocs = map_parts(nz_allocation,a.counters)
  DistributedAllocationCOO(allocs,a.rows,a.cols)
end

struct DistributedAllocationCOO{A,B,C} <:GridapType
  allocs::A
  rows::B
  cols::C
  function DistributedAllocationCOO(
    allocs::AbstractPData{<:Algebra.AllocationCOO},
    rows::PRange,
    cols::PRange)
    A = typeof(allocs)
    B = typeof(rows)
    C = typeof(cols)
    new{A,B,C}(allocs,rows,cols)
  end
end

function local_views(a::DistributedAllocationCOO)
  a.allocs
end

function Algebra.create_from_nz(a::DistributedAllocationCOO)
  A, = _create_from_nz_with_callback(i->nothing,a)
  A
end

function _create_from_nz_with_callback(callback,a::DistributedAllocationCOO)

  # TODO for the moment we assume that we integrate
  # on owned cells only and thus we need to send some rows
  # to neighbor procs.
  # If we integrate also on ghost cells then this simplifies a lot!

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
  first_gdof(ids) = num_oids(ids)>0 ? Int(ids.lid_to_gid[ids.oid_to_lid[1]]) : 1
  first_grdof = map_parts(first_gdof,rdofs.partition)
  first_gcdof = map_parts(first_gdof,cdofs.partition)
  rneigs_snd = rdofs.exchanger.parts_snd
  rneigs_rcv = rdofs.exchanger.parts_rcv
  cneigs_snd = cdofs.exchanger.parts_snd
  cneigs_rcv = cdofs.exchanger.parts_rcv

  # convert I and J to global dof ids
  to_gids!(I,rdofs)
  to_gids!(J,cdofs)

  # Find the ghost rows
  hrow_to_hrdof = touched_hids(rdofs,I)
  function find_gid_and_part(hid_to_hdof,dofs)
    hid_to_ldof = view(dofs.hid_to_lid,hid_to_hdof)
    hid_to_gid = dofs.lid_to_gid[hid_to_ldof]
    hid_to_part = dofs.lid_to_part[hid_to_ldof]
    hid_to_gid, hid_to_part
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

  # Convert again I,J to local numeration
  to_lids!(I,rows)
  to_lids!(J,cols)

  # Compress the local matrices
  values = map_parts(create_from_nz,a.allocs)

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

struct PVectorBuilder{T}
  local_vector_type::Type{T}
end

function Algebra.nz_counter(builder::PVectorBuilder,axs::Tuple{<:PRange})
  T = builder.local_vector_type
  rows, = axs
  counters = map_parts(rows.partition) do rows
    axs = (Base.OneTo(num_lids(rows)),)
    nz_counter(ArrayBuilder(T),axs)
  end
  PVectorCounter(counters,rows)
end

struct PVectorCounter{A,B}
  counters::A
  rows::B
end

Algebra.LoopStyle(::Type{<:PVectorCounter}) = DoNotLoop()

function local_views(a::PVectorCounter)
  a.counters
end

function Arrays.nz_allocation(a::PVectorCounter)
  dofs = a.rows
  values = map_parts(nz_allocation,a.counters)
  PVector(values,dofs)
end

function Algebra.create_from_nz(a::PVector)
  @notimplemented
end

function Algebra.create_from_nz(a::DistributedAllocationCOO,b_fespace::PVector)

  # TODO for the moment we assume that we integrate
  # on owned cells only and thus we need to send some rows
  # to neighbor procs.
  # If we integrate also on ghost cells then this simplifies a lot!
  
  # The ghost values in b_fespace are aligned with the FESpace
  # but not with the ghost values in the rows of A

  A,b = _create_from_nz_with_callback(a) do rows

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

  # now we can assemble contributions
  assemble!(b)

  A,b
end

