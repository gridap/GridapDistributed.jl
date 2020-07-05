# Assembly related

struct SequentialIJV{A,B}
  dIJV::A
  gIJV::B
end

get_distributed_data(a::SequentialIJV) = a.dIJV

function Gridap.Algebra.allocate_vector(::Type{V},gids::DistributedIndexSet) where V <: AbstractVector
  ngids = num_gids(gids)
  allocate_vector(V,ngids)
end

function allocate_local_vector(
  strat::Union{DistributedAssemblyStrategy{RowsComputedLocally{false}},DistributedAssemblyStrategy{OwnedCellsStrategy{false}}},
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
  strat::Union{DistributedAssemblyStrategy{RowsComputedLocally{true}},DistributedAssemblyStrategy{OwnedCellsStrategy{true}}},
  ::Type{V},
  indices::SequentialDistributedIndexSet,
) where V<:Vector
   T = get_local_vector_type(V)
   vec=T(undef,num_gids(indices))
   fill!(vec,zero(eltype(T)))
   vec
end


function assemble_global_matrix(strat::Union{DistributedAssemblyStrategy{RowsComputedLocally{T}},DistributedAssemblyStrategy{OwnedCellsStrategy{T}}},
                                ::Type{M},
                                IJV::SequentialIJV,
                                m::DistributedIndexSet,
                                n::DistributedIndexSet) where {T,M}
  if (!T)
     do_on_parts(IJV.dIJV,m,n) do part, IJV, mindexset, nindexset
        I,J,V = IJV
        for i=1:length(I)
          I[i]=mindexset.lid_to_gid[I[i]]
          J[i]=nindexset.lid_to_gid[J[i]]
        end
     end
  end
  I,J,V = IJV.gIJV
  A=sparse_from_coo(M,I,J,V,num_gids(m),num_gids(n))
end

function assemble_global_vector(strat::Union{DistributedAssemblyStrategy{RowsComputedLocally{false}},DistributedAssemblyStrategy{OwnedCellsStrategy{false}}},
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



function assemble_global_vector(strat::Union{DistributedAssemblyStrategy{RowsComputedLocally{true}},DistributedAssemblyStrategy{OwnedCellsStrategy{true}}},
                                ::Type{M},
                                b::M,
                                m::DistributedIndexSet) where M <: Vector
  b
end

function Gridap.Algebra.allocate_coo_vectors(::Type{M},dn::DistributedData) where M <: AbstractMatrix

  part_to_n = gather(dn)
  n = sum(part_to_n)
  gIJV = allocate_coo_vectors(M,n)

  _fill_offsets!(part_to_n)

  dIJV = DistributedData(get_comm(dn), part_to_n) do part, part_to_n
    spos=part_to_n[part]+1
    if (part == length(part_to_n))
      epos=n
    else
      epos=part_to_n[part+1]
    end
    map( i -> SubVector(i,spos,epos), gIJV)
  end

  SequentialIJV(dIJV,gIJV)
end

function Gridap.Algebra.finalize_coo!(
  ::Type{M},IJV::SequentialIJV,m::DistributedIndexSet,n::DistributedIndexSet) where M <: AbstractMatrix
  I,J,V = IJV.gIJV
  finalize_coo!(M,I,J,V,num_gids(m),num_gids(n))
end
