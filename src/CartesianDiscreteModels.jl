
function Gridap.CartesianDiscreteModel(comm::Communicator,subdomains::Tuple,args...)
  desc = CartesianDescriptor(args...)
  CartesianDiscreteModel(comm,subdomains,desc)
end

function Gridap.CartesianDiscreteModel(
  comm::Communicator,subdomains::Tuple,gdesc::CartesianDescriptor{D,T,F}) where {D,T,F}

  nsubdoms = prod(subdomains)
  ngcells = prod(Tuple(gdesc.partition))

  function init_models(isubdom)
    ldesc = local_cartesian_descriptor(gdesc,subdomains,isubdom)
    CartesianDiscreteModel(ldesc)
  end

  function init_gids(isubdom)
    lid_to_gid, lid_to_owner = local_cartesian_gids(gdesc,subdomains,isubdom)
    GhostedVectorPart(lid_to_gid,lid_to_gid,lid_to_owner)
  end

  S = CartesianDiscreteModel{D,T,F}
  models = ScatteredVector{S}(comm,nsubdoms,init_models)
  gids = GhostedVector{Int}(comm,ngcells,nsubdoms,init_gids)

  DistributedDiscreteModel(models,gids)
end

function local_cartesian_descriptor_1d(
  gdesc::CartesianDescriptor{1},nsubdoms::Integer,isubdom::Integer)

  gcells, = gdesc.partition
  gorigin, = gdesc.origin
  h, = gdesc.sizes
  H = h*gcells/nsubdoms

  orange = uniform_partition_1d(gcells,nsubdoms,isubdom)
  ocells = length(orange)

  if isubdom == 1
    lcells =  ocells + 1
    lorigin = gorigin 
  elseif isubdom != nsubdoms
    lcells = ocells + 2
    lorigin = gorigin + H*(isubdom-1)-h
  else
    lcells = ocells + 1
    lorigin = gorigin + H*(isubdom-1)-h
  end

  CartesianDescriptor(lorigin,h,lcells,gdesc.map)

end

function local_cartesian_gids_1d(
  gdesc::CartesianDescriptor{1},nsubdoms::Integer,isubdom::Integer)

  gcells, = gdesc.partition

  orange = uniform_partition_1d(gcells,nsubdoms,isubdom)
  ocells = length(orange)

  if isubdom == 1
    lrange = orange.start:(orange.stop+1)
  elseif isubdom != nsubdoms
    lrange = (orange.start-1):(orange.stop+1)
  else
    lrange = (orange.start-1):orange.stop
  end

  lcells = length(lrange)
  lid_to_gid = collect(Int,lrange)
  lid_to_owner = fill(isubdom,lcells)

  if isubdom == 1
    lid_to_owner[end] = 2
  elseif isubdom != nsubdoms
    lid_to_owner[1] = isubdom - 1
    lid_to_owner[end] = isubdom + 1
  else
    lid_to_owner[1] = isubdom - 1
  end

  lid_to_gid, lid_to_owner
end

function local_cartesian_descriptor(gdesc::CartesianDescriptor,nsubdoms::Tuple,isubdom::Integer)
  cis = CartesianIndices(nsubdoms)
  ci = cis[isubdom]
  local_cartesian_descriptor(gdesc,nsubdoms,ci)
end

function local_cartesian_descriptor(
  gdesc::CartesianDescriptor{D,T},nsubdoms::Tuple,isubdom::CartesianIndex) where {D,T}

  origin = zeros(T,D)
  sizes = zeros(T,D)
  partition = zeros(Int,D)
  for d in 1:D
    gdesc_d = CartesianDescriptor(gdesc.origin[d],gdesc.sizes[d],gdesc.partition[d])
    ldesc_d = local_cartesian_descriptor_1d(gdesc_d,nsubdoms[d],isubdom[d])
    origin[d] = ldesc_d.origin[1]
    sizes[d] = ldesc_d.sizes[1]
    partition[d] = ldesc_d.partition[1]
  end

  CartesianDescriptor(origin,sizes,partition,gdesc.map)
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
    isowned = true
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

