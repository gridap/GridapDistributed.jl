module GridapDistributedTests

using GridapDistributed
using Test

abstract type DistributionStyle end

struct SequentialDistribution <: DistributionStyle end

abstract type ScatteredVector{T} end

function ScatteredVector{T}(::DistributionStyle,workers,glength,fun) where T
end

struct SequentialScatteredVector{T} <: ScatteredVector{T}
  data::Vector{T}
end

function ScatteredVector{T}(::SequentialDistribution,workers,glength,fun) where T
  data = [fun(T,glength,i) for i in 1:glength]
  SequentialScatteredVector(data)
end

abstract type GhostedVector{T} end

struct GhostedVectorPart{T}
  lid_to_item::Vector{T}
  lid_to_gid::Vector{Int}
  lid_to_isowned::Vector{Bool}
end

function GhostedVector{T}(::DistributionStyle,workers,glength,fun,nparts) where T
end

function GhostedVector{T}(s::DistributionStyle,workers,glength,fun) where T
  GhostedVector{T}(s,workers,glength,fun,length(workers))
end

struct SequentialGhostedVector{T} <: GhostedVector{T}
  data::Vector{GhostedVectorPart{T}}
end

function GhostedVector{T}(::SequentialDistribution,workers,glength,fun,nparts) where T
  data = [ fun(T,glength,nparts,i) for i in 1:nparts ]
  SequentialGhostedVector{T}(data)
end

using Gridap
using Gridap.Geometry
using Gridap.TensorValues: mutable

function uniform_ghosted_partition_1d(glength,np,pid)

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

  if pid == 1
    llength = olength + 1
    lid_to_gid = collect(Int, (1+offset):(olength+offset+1) )
    lid_to_isowned = fill(true,llength)
    lid_to_isowned[end] = false
  elseif pid != np
    llength = olength + 2
    lid_to_gid = collect(Int, (1+offset-1):(olength+offset+1) )
    lid_to_isowned = fill(true,llength)
    lid_to_isowned[1] = false
    lid_to_isowned[end] = false
  else
    llength = olength + 1
    lid_to_gid = collect(Int, (1+offset-1):(olength+offset) )
    lid_to_isowned = fill(true,llength)
    lid_to_isowned[1] = false
  end

  lid_to_gid, lid_to_isowned

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

struct DistributedDiscreteModel
  models::ScatteredVector{<:DiscreteModel}
  gids::GhostedVector{Int}
end

function Gridap.CartesianDiscreteModel(s::DistributionStyle,workers,subdomains,args...)
  desc = CartesianDescriptor(args...)
  CartesianDiscreteModel(s,workers,subdomains,desc)
end

function Gridap.CartesianDiscreteModel(
  s::DistributionStyle,workers,subdomains,desc::CartesianDescriptor{D,T,F}) where {D,T,F}

  gcells = Tuple(desc.partition)
  dx = Point(map((s,n)-> s*n,Tuple(desc.sizes),gcells))
  gpmin = desc.origin
  gpmax = gpmin + dx

  function f1(T,glength,i) 
    cis = CartesianIndices(subdomains)
    pid = cis[i]
    lpmin, lpmax = uniform_ghosted_subdomains_nd(gpmin,gpmax,gcells,subdomains,pid)
    lcells = map( (n,np,p) -> length(uniform_partition_1d(n,np,p)) , gcells, subdomains, Tuple(pid) )
    @show lcells
    @show lpmin
    @show lpmax
    CartesianDiscreteModel(lpmin,lpmax,lcells)
  end

  S = CartesianDiscreteModel{D,T,F}
  glength = prod(subdomains)
  models = ScatteredVector{S}(s,workers,glength,f1)

  function f2(T,glength,nparts,i) 
    cis = CartesianIndices(subdomains)
    pid = cis[i]
    lid_to_gid, lid_to_isowned = uniform_ghosted_partition_nd(gcells,subdomains,pid)
    p = GhostedVectorPart{T}(lid_to_gid,lid_to_gid,lid_to_isowned)
    @show p
    GhostedVectorPart{T}(lid_to_gid,lid_to_gid,lid_to_isowned)
  end

  glength = prod(gcells)
  nparts = prod(subdomains)
  gids = GhostedVector{Int}(s,workers,glength,f2,nparts)

  DistributedDiscreteModel(models,gids)

end

function uniform_ghosted_partition_nd(glength::Tuple,np::Tuple,pid::CartesianIndex)
  D = length(glength)

  d_to_data = [uniform_ghosted_partition_1d(glength[d],np[d],pid[d]) for d in 1:D]

  d_to_llength = map( data -> length(data[1]),d_to_data)

  lcis = CartesianIndices(Tuple(d_to_llength))
  llis = LinearIndices(lcis)
  gcis = CartesianIndices(glength)
  glis = LinearIndices(gcis)

  lid_to_gid = zeros(Int,length(lcis))
  lid_to_isowned = zeros(Bool,length(lcis))
  _gci = zeros(Int,D)

  for lci in lcis

    isowned = true
    for d in 1:D
      lid_to_gid_d, lid_to_isowned_d = d_to_data[d]
      _gci[d] = lid_to_gid_d[lci[d]]
      isowned = isowned && lid_to_isowned_d[lci[d]]
    end
    lid = llis[lci]
    gci = CartesianIndex(Tuple(_gci))
    lid_to_gid[lid] = glis[gci]
    lid_to_isowned[lid] = isowned

  end

  lid_to_gid, lid_to_isowned
end

function uniform_ghosted_subdomains_1d(gpmin,gpmax,gcells,np,pid)
  h = (gpmax - gpmin) / gcells
  H = (gpmax - gpmin) / np
  if pid == 1
    lpmin = gpmin
    lpmax = gpmin + H + h
  elseif pid !=np
    lpmin = gpmin + H*(pid-1) - h
    lpmax = gpmin + H*(pid) + h
  else
    lpmin = gpmin + H*(pid-1) -h
    lpmax = gpmax
  end
  lpmin, lpmax
end

function uniform_ghosted_subdomains_nd(gpmin::Point,gpmax::Point,gcells::Tuple,np::Tuple,pid::CartesianIndex)
  D = length(gcells)
  T = typeof(gpmin[1]/np[1])
  lpmin = zero(mutable(Point{D,T}))
  lpmax = zero(mutable(Point{D,T}))
  for d in 1:D
    lpmin_d, lpmax_d = uniform_ghosted_subdomains_1d(gpmin[d],gpmax[d],gcells[d],np[d],pid[d])
    lpmin[d] = lpmin_d
    lpmax[d] = lpmax_d
  end
  Point(lpmin), Point(lpmax)
end

function Gridap.writevtk(model::DistributedDiscreteModel,filebase::String)
  for (i,model_i) in enumerate(model.models.data)
    isowned = collect(Int,model.gids.data[i].lid_to_isowned)
    writevtk(Triangulation(model_i),"$(filebase)_$(i)",celldata=["isowned"=>isowned])
  end
end

gpmin = 10
gpmax = 20
glength = 10
np = 5
for pid in 1:np
  lid_to_gid, lid_to_isowned = uniform_ghosted_partition_1d(glength,np,pid)
  lpmin, lpmax = uniform_ghosted_subdomains_1d(gpmin,gpmax,glength,np,pid)
  #@show pid
  #@show lid_to_gid
  #@show lid_to_isowned
  #@show lpmin
  #@show lpmax
end


glength = (4,4)
np = (2,2)
gpmin = Point(10,0)
gpmax = Point(20,10)
for pid in CartesianIndices(np)
  lid_to_gid, lid_to_isowned = uniform_ghosted_partition_nd(glength,np,pid)
  lpmin, lpmax = uniform_ghosted_subdomains_nd(gpmin,gpmax,glength,np,pid)
  @show pid
  @show lid_to_gid
  @show lid_to_isowned
  @show lpmin
  @show lpmax
end

s = SequentialDistribution()
workers = [1]
subdomains = (2,2)
domain = (0,1,0,1)
cells = (3,3)

model = CartesianDiscreteModel(s,workers,subdomains,domain,cells)

writevtk(model,"model")


end # module
