function Gridap.CartesianDiscreteModel(comm::Communicator,subdomains::Tuple,args...)
  desc = CartesianDescriptor(args...)
  CartesianDiscreteModel(comm,subdomains,desc)
end

function Gridap.CartesianDiscreteModel(
  comm::Communicator,subdomains::Tuple,gdesc::CartesianDescriptor)

  nsubdoms = prod(subdomains)
  ngcells = prod(Tuple(gdesc.partition))
  models = DistributedData(comm) do isubdom
    cmin, cmax = compute_cmin_cmax(gdesc, subdomains, isubdom)
    CartesianDiscreteModel(gdesc, cmin, cmax)
  end

  gids = DistributedIndexSet(comm,ngcells) do isubdom
    lid_to_gid, lid_to_owner = local_cartesian_gids(gdesc,subdomains,isubdom)
    IndexSet(ngcells,lid_to_gid,lid_to_owner)
  end

  DistributedDiscreteModel(models,gids)
end

function compute_cmin_cmax(
  gdesc::CartesianDescriptor{D,T},
  nsubdoms::Tuple,
  isubdom::Integer,
) where {D,T}
  cis = CartesianIndices(nsubdoms)
  ci = cis[isubdom]
  cmin = Vector{Int}(undef, D)
  cmax = Vector{Int}(undef, D)
  for d = 1:D
    orange = uniform_partition_1d(gdesc.partition[d], nsubdoms[d], ci[d])
    lrange = extend_range_with_ghost_cells(orange, nsubdoms[d], ci[d])
    cmin[d] = first(lrange)
    cmax[d] = last(lrange)
  end
  return CartesianIndex(Tuple(cmin)), CartesianIndex(Tuple(cmax))
end

function local_cartesian_gids_1d(
  gdesc::CartesianDescriptor{1},
  nsubdoms::Integer,
  isubdom::Integer,
)
  gcells, = gdesc.partition
  orange = uniform_partition_1d(gcells, nsubdoms, isubdom)
  lrange = extend_range_with_ghost_cells(orange, nsubdoms, isubdom)
  lcells = length(lrange)
  lid_to_gid = collect(Int, lrange)
  lid_to_owner = fill(isubdom, lcells)
  if nsubdoms == 1
    nothing
  elseif isubdom == 1
    lid_to_owner[end] = 2
  elseif isubdom != nsubdoms
    lid_to_owner[1] = isubdom - 1
    lid_to_owner[end] = isubdom + 1
  else
    lid_to_owner[1] = isubdom - 1
  end
  lid_to_gid, lid_to_owner
end

function extend_range_with_ghost_cells(orange, nsubdoms, isubdom)
  if nsubdoms == 1
    lrange = orange
  elseif isubdom == 1
    lrange = orange.start:(orange.stop+1)
  elseif isubdom != nsubdoms
    lrange = (orange.start-1):(orange.stop+1)
  else
    lrange = (orange.start-1):orange.stop
  end
  return lrange
end

function local_cartesian_gids(
  gdesc::CartesianDescriptor{D},nsubdoms::Tuple,isubdom::Integer) where D
  cis = CartesianIndices(nsubdoms)
  ci = cis[isubdom]
  local_cartesian_gids(gdesc,nsubdoms,ci)
end

function local_cartesian_gids(
  gdesc::CartesianDescriptor{D},nsubdoms::Tuple,isubdom::CartesianIndex) where D

  d_to_lid_to_gid = Vector{Int}[]
  d_to_lid_to_owner = Vector{Int}[]
  for d in 1:D
    gdesc_d = CartesianDescriptor(gdesc.origin[d],gdesc.sizes[d],gdesc.partition[d])
    lid_to_gid_d, lid_to_owner_d = local_cartesian_gids_1d(gdesc_d,nsubdoms[d],isubdom[d])
    push!(d_to_lid_to_gid,lid_to_gid_d)
    push!(d_to_lid_to_owner,lid_to_owner_d)
  end

  d_to_llength = Tuple(map(length,d_to_lid_to_gid))
  d_to_glength = Tuple(gdesc.partition)

  lcis = CartesianIndices(d_to_llength)
  gcis = CartesianIndices(d_to_glength)
  scis = CartesianIndices(nsubdoms)
  llis = LinearIndices(lcis)
  glis = LinearIndices(gcis)
  slis = LinearIndices(scis)

  lid_to_gid = zeros(Int,length(lcis))
  lid_to_owner = zeros(Int,length(lcis))
  gci = zeros(Int,D)
  sci = zeros(Int,D)

  for lci in lcis
    for d in 1:D
      gci[d] = d_to_lid_to_gid[d][lci[d]]
      sci[d] = d_to_lid_to_owner[d][lci[d]]
    end
    lid = llis[lci]
    lid_to_gid[lid] = glis[CartesianIndex(Tuple(gci))]
    lid_to_owner[lid] = slis[CartesianIndex(Tuple(sci))]
  end

  lid_to_gid, lid_to_owner
end

function uniform_partition_1d(glength,np,pid)
  _olength = glength ÷ np
  _offset = _olength * (pid-1)
  _rem = glength % np
  if _rem < (np-pid+1)
    olength = _olength
    offset = _offset
  else
    olength = _olength + 1
    offset = _offset + pid - (np-_rem) - 1
  end
  (1+offset):(olength+offset)
end
