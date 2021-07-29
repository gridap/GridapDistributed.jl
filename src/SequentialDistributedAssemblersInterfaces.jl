# Assembly related

function default_assembly_strategy_type(::SequentialCommunicator)
  OwnedAndGhostCellsAssemblyStrategy
end

function default_map_dofs_type(::SequentialCommunicator)
  MapDoFsTypeGlobal
end


function Gridap.Algebra.allocate_vector(::Type{V},gids::DistributedIndexSet) where V <: AbstractVector
  ngids = num_gids(gids)
  allocate_vector(V,ngids)
end

function allocate_local_vector(
  strat::Union{DistributedAssemblyStrategy{OwnedAndGhostCellsAssemblyStrategy{MapDoFsTypeProcLocal}},DistributedAssemblyStrategy{OwnedCellsAssemblyStrategy{MapDoFsTypeProcLocal}}},
  ::Type{V},
  indices::SequentialDistributedIndexSet,
) where V<:Vector
  DistributedData(indices) do part,index
   T = get_local_vector_type(V)
   lvec=T(undef,length(index.lid_to_gid))
   fill!(lvec,zero(eltype(T)))
   lvec
  end
end

function allocate_local_vector(
  strat::Union{DistributedAssemblyStrategy{OwnedAndGhostCellsAssemblyStrategy{MapDoFsTypeGlobal}},DistributedAssemblyStrategy{OwnedCellsAssemblyStrategy{MapDoFsTypeGlobal}}},
  ::Type{V},
  indices::SequentialDistributedIndexSet,
) where V<:Vector
   T = get_local_vector_type(V)
   vec=T(undef,num_gids(indices))
   fill!(vec,zero(eltype(T)))
   vec
end


function assemble_global_matrix(strat::Union{DistributedAssemblyStrategy{OwnedAndGhostCellsAssemblyStrategy{T}},DistributedAssemblyStrategy{OwnedCellsAssemblyStrategy{T}}},
                                ::Type{M},
                                dIJV::SequentialDistributedData,
                                m::DistributedIndexSet,
                                n::DistributedIndexSet) where {T,M}
  if (T==MapDoFsTypeProcLocal)
     do_on_parts(dIJV,m,n) do part, IJV, mindexset, nindexset
        I,J,V = IJV
        for i=1:length(I)
          I[i]=mindexset.lid_to_gid[I[i]]
          J[i]=nindexset.lid_to_gid[J[i]]
        end
     end
  end
  if (length(dIJV.parts)==1)
    I,J,V=dIJV.parts[1]
  else
    I=lazy_append(dIJV.parts[1][1],dIJV.parts[2][1])
    J=lazy_append(dIJV.parts[1][2],dIJV.parts[2][2])
    V=lazy_append(dIJV.parts[1][3],dIJV.parts[2][3])
    for part=3:length(dIJV.parts)
      I=lazy_append(I,dIJV.parts[part][1])
      J=lazy_append(J,dIJV.parts[part][2])
      V=lazy_append(V,dIJV.parts[part][3])
    end
  end
  A=sparse_from_coo(M,I,J,V,num_gids(m),num_gids(n))
end

function assemble_global_vector(strat::Union{DistributedAssemblyStrategy{OwnedAndGhostCellsAssemblyStrategy{MapDoFsTypeProcLocal}},DistributedAssemblyStrategy{OwnedCellsAssemblyStrategy{MapDoFsTypeProcLocal}}},
                                ::Type{M},
                                db::DistributedData,
                                m::DistributedIndexSet) where M <: Vector
  b=allocate_vector(M, num_gids(m))
  do_on_parts(m, db, b) do part, mindexset, blocal, b
    for i=1:length(blocal)
      b[mindexset.lid_to_gid[i]] += blocal[i]
    end
  end
  b
end

function assemble_global_vector(strat::Union{DistributedAssemblyStrategy{OwnedAndGhostCellsAssemblyStrategy{MapDoFsTypeGlobal}},DistributedAssemblyStrategy{OwnedCellsAssemblyStrategy{MapDoFsTypeGlobal}}},
                                ::Type{M},
                                b::M,
                                m::DistributedIndexSet) where M <: Vector
  b
end

function Gridap.Algebra.finalize_coo!(
  ::Type{M},
  IJV::SequentialDistributedData,
  m::DistributedIndexSet,
  n::DistributedIndexSet) where M <: AbstractMatrix
  for part in IJV.parts
    finalize_coo!(M,part[1],part[2],part[3],num_gids(m),num_gids(n))
  end
end
