# Specializations
struct MPIPETScDistributedVector{T <: Union{Number,AbstractVector{<:Number}},V <: AbstractVector{T},A,B,C} <: DistributedVector{T}
    part::V
    indices::MPIPETScDistributedIndexSet{A,B,C}
    vecghost::PETSc.Vec{Float64}
end

get_comm(a::MPIPETScDistributedVector) = get_comm(a.indices)

get_part(
  comm::MPIPETScCommunicator,
  a::MPIPETScDistributedVector,
  part::Integer) = a.part

get_part(
  comm::MPIPETScCommunicator,
  a::PETSc.Vec{Float64},
  part::Integer) = a


function DistributedVector(
  initializer::Function, indices::MPIPETScDistributedIndexSet,args...)
    comm = get_comm(indices)
    data = DistributedData(initializer, comm, args...)
    part = data.part
    V = typeof(part)
    if (V <: Table)
        sizes = [part.ptrs[i + 1] - part.ptrs[i]
                     for i = 1:length(part.ptrs)-1]
        max_sizes = maximum(sizes)
        min_sizes = minimum(sizes)
        if (max_sizes == min_sizes)
            scalar_indices =
              _create_fixed_length_indices(max_sizes, indices)
        else
            scalar_indices =
              _create_variable_length_indices(sizes, indices)
        end
        T=get_data_eltype(V)
        @assert sizeof(T) == sizeof(Float64)
        petsc_ghost_array = Vector{Float64}(undef, length(part.data))
        new_data = reinterpret(T, petsc_ghost_array)

        current=1
        p_reindex_data=similar(part.data,Int32)
        for i=1:length(indices.petsc_to_app_locidx)
           j=indices.petsc_to_app_locidx[i]
           size=part.ptrs[j+1]-part.ptrs[j]
           new_data[current:current+size-1]=
              part.data[part.ptrs[j]:part.ptrs[j+1]-1]
           p_reindex_data[part.ptrs[j]:part.ptrs[j+1]-1] .=
                           current:current+size-1
           current=current+size
        end
        new_data_reindexed=reindex(new_data,p_reindex_data)
        new_part=Table(new_data_reindexed,part.ptrs)
        vecghost = create_ghost_vector(petsc_ghost_array, scalar_indices)
        MPIPETScDistributedVector(new_part, scalar_indices, vecghost)
    elseif (V <: Vector{<:Number})
        T = eltype(V)
        @assert sizeof(T) == sizeof(Float64)
        petsc_ghost_array = Vector{Float64}(undef, length(part))
        new_part = reindex(reinterpret(T, petsc_ghost_array),
                                  indices.app_to_petsc_locidx)
        copy_entries!(new_part, part)
        vecghost = create_ghost_vector(petsc_ghost_array, indices)
        MPIPETScDistributedVector(new_part, indices, vecghost)
    else
        @error "Initializer function returns unsupported local vector type"
    end

end

function _create_fixed_length_indices(
     length_entry::Int,
     indices::MPIPETScDistributedIndexSet)
    l = length_entry
    n = l * indices.ngids
    indices = DistributedIndexSet(get_comm(indices), n, indices, l, n) do part, indices, l, n
        lid_to_gid   = Vector{Int}(undef, l * length(indices.lid_to_owner))
        lid_to_owner = Vector{Int}(undef, l * length(indices.lid_to_owner))
        k = 1
        for i = 1:length(indices.lid_to_owner)
            offset = (indices.lid_to_gid[i] - 1) * l
            for j = 1:l
                lid_to_gid[k]   = offset + j
                lid_to_owner[k] = indices.lid_to_owner[i]
                k = k + 1
            end
        end
        IndexSet(n, lid_to_gid, lid_to_owner)
    end
    indices
end

function _create_variable_length_indices(
  length_entries::AbstractVector{<:Integer},
  indices::MPIPETScDistributedIndexSet,
)

# println(T)
# println(eltype(T))
# @assert sizeof(eltype(T)) == sizeof(Float64)
    comm      = get_comm(indices)
    part      = MPI.Comm_rank(comm.comm) + 1
    num_owned_entries = 0
    num_local_entries = 0
    lid_to_owner = indices.parts.part.lid_to_owner
    for i = 1:length(lid_to_owner)
        if (lid_to_owner[i] == part)
            num_owned_entries = num_owned_entries + length_entries[i]
        end
        num_local_entries = num_local_entries + length_entries[i]
    end

    sndbuf = Ref{Int64}(num_owned_entries)
    rcvbuf = Ref{Int64}()

    MPI.Exscan!(sndbuf, rcvbuf, 1, +, comm.comm)
    (part == 1) ? (offset = 1) : (offset = rcvbuf[] + 1)

    sendrecvbuf = Ref{Int64}(num_owned_entries)
    MPI.Allreduce!(sendrecvbuf, +, comm.comm)
    n = sendrecvbuf[]

    offsets = DistributedVector(indices, indices) do part, indices
        local_part = Vector{Int}(undef,length(indices.lid_to_owner))
        for i = 1:length(indices.lid_to_owner)
            if (indices.lid_to_owner[i] == part)
                local_part[i] = offset
                for j = 1:length_entries[i]
                    offset = offset + 1
                end
            end
        end
        local_part
    end
    exchange!(offsets)

    indices = DistributedIndexSet(get_comm(indices), n, indices, offsets) do part, indices, offsets
        lid_to_gid   = Vector{Int}(undef, num_local_entries)
        lid_to_owner = Vector{Int}(undef, num_local_entries)
        k = 1
        for i = 1:length(indices.lid_to_owner)
            offset = offsets[i]
            for j = 1:length_entries[i]
                lid_to_gid[k]   = offset
                lid_to_owner[k] = indices.lid_to_owner[i]
                k = k + 1
                offset = offset + 1
            end
        end
        IndexSet(n, lid_to_gid, lid_to_owner)
    end
    indices
end

function unpack_all_entries!(a::MPIPETScDistributedVector{T}) where T
    num_local = length(a.indices.app_to_petsc_locidx)
    lvec = PETSc.LocalVector(a.vecghost, num_local)
    _unpack_all_entries!(T, a.part, a.indices.app_to_petsc_locidx, lvec)
    PETSc.restore(lvec)
end

function Base.getindex(a::PETSc.Vec{Float64}, indices::MPIPETScDistributedIndexSet)
    result = DistributedVector(indices,indices) do part, indices
       Vector{Float64}(undef,length(indices.lid_to_owner))
    end
    copy!(result.vecghost, a)
    exchange!(result.vecghost)
    result
end

function exchange!(a::PETSc.Vec{T}) where T
  # Send data
    PETSc.scatter!(a)
end

function exchange!(a::MPIPETScDistributedVector{T}) where T
    indices = a.indices
    local_part = a.part
    lid_to_owner = indices.parts.part.lid_to_owner
    petsc_to_app_locidx = indices.petsc_to_app_locidx
    app_to_petsc_locidx = indices.app_to_petsc_locidx
    comm = get_comm(indices)
    comm_rank = MPI.Comm_rank(comm.comm)

  # Pack data
  # _pack_local_entries!(a.vecghost, local_part, lid_to_owner, comm_rank)

    exchange!(a.vecghost)

  #  Unpack data
  # num_local = length(lid_to_owner)
  # lvec = PETSc.LocalVector(a.vecghost,num_local)
  # _unpack_ghost_entries!(eltype(local_part), local_part, lid_to_owner, comm_rank, app_to_petsc_locidx, lvec)
  # PETSc.restore(lvec)
end

function _pack_local_entries!(vecghost, local_part, lid_to_owner, comm_rank)
    num_owned = PETSc.lengthlocal(vecghost)
    idxs = Vector{Int}(undef, num_owned)
    vals = Vector{Float64}(undef, num_owned)
  # Pack data
    k = 1
    current = 1
    for i = 1:length(local_part)
        for j = 1:length(local_part[i])
            if ( lid_to_owner[k] == comm_rank + 1 )
                idxs[current] = current - 1
                vals[current] = reinterpret(Float64, local_part[i][j])
                current = current + 1
            end
            k = k + 1
        end
    end
    set_values_local!(vecghost, idxs, vals, PETSc.C.INSERT_VALUES)
    AssemblyBegin(vecghost, PETSc.C.MAT_FINAL_ASSEMBLY)
    AssemblyEnd(vecghost, PETSc.C.MAT_FINAL_ASSEMBLY)
end

function _pack_all_entries!(vecghost, local_part)
    num_local = prod(size(local_part))
    idxs = Vector{Int}(undef, num_local)
    vals = Vector{Float64}(undef, num_local)

  # Pack data
    k = 1
    current = 1
    for i = 1:length(local_part)
        for j = 1:length(local_part[i])
            idxs[current] = current - 1
            vals[current] = reinterpret(Float64, local_part[i][j])
            current = current + 1
            k = k + 1
        end
    end
    set_values_local!(vecghost, idxs, vals, PETSc.C.INSERT_VALUES)
    AssemblyBegin(vecghost, PETSc.C.MAT_FINAL_ASSEMBLY)
    AssemblyEnd(vecghost, PETSc.C.MAT_FINAL_ASSEMBLY)
end

function _unpack_ghost_entries!(
  T::Type{<:Number},
  local_part,
  lid_to_owner,
  comm_rank,
  app_to_petsc_locidx,
  lvec,
)
    for i = 1:length(local_part)
        if (lid_to_owner[i] != comm_rank + 1)
            local_part[i] = reinterpret(T, lvec.a[app_to_petsc_locidx[i]])
        end
    end
end

function _unpack_ghost_entries!(
  T::Type{<:AbstractVector{K}},
  local_part,
  lid_to_owner,
  comm_rank,
  app_to_petsc_locidx,
  lvec,
) where K <: Number
    k = 1
    for i = 1:length(local_part)
        for j = 1:length(local_part[i])
            if (lid_to_owner[k] != comm_rank + 1)
                local_part[i][j] = reinterpret(K, lvec.a[app_to_petsc_locidx[k]])
            end
            k = k + 1
        end
    end
end

function _unpack_all_entries!(
  T::Type{<:Number},
  local_part,
  app_to_petsc_locidx,
  lvec,
)
    for i = 1:length(local_part)
        local_part[i] = reinterpret(T, lvec.a[app_to_petsc_locidx[i]])
    end
end

function _unpack_all_entries!(
  T::Type{<:AbstractVector{K}},
  local_part,
  app_to_petsc_locidx,
  lvec,
) where {K <: Number}
    k = 1
    for i = 1:length(local_part)
        for j = 1:length(local_part[i])
            local_part[i][j] = reinterpret(K, lvec.a[app_to_petsc_locidx[k]])
            k = k + 1
        end
    end
end
