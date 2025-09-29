
# DistributedCellPoint
"""
"""
struct DistributedCellPoint{A<:AbstractArray{<:CellPoint},B<:DistributedTriangulation} <: CellDatum
  points::A
  trian ::B
end

local_views(a::DistributedCellPoint) = a.points
CellData.get_triangulation(a::DistributedCellPoint) = a.trian

function CellData.DomainStyle(::Type{<:DistributedCellPoint{A}}) where A 
  DomainStyle(eltype(A))
end

# DistributedCellField
"""
"""
struct DistributedCellField{A,B,C} <: CellField
  fields::A
  trian ::B
  metadata::C
  function DistributedCellField(
    fields::AbstractArray{<:CellField},
    trian ::DistributedTriangulation,
    metadata=nothing)

    A = typeof(fields)
    B = typeof(trian)
    C = typeof(metadata)
    new{A,B,C}(fields,trian,metadata)
  end
end

local_views(a::DistributedCellField) = a.fields
CellData.get_triangulation(a::DistributedCellField) = a.trian

function CellData.DomainStyle(::Type{<:DistributedCellField{A}}) where A 
  DomainStyle(eltype(A))
end

# Constructors

function CellData.CellField(f::Function,trian::DistributedTriangulation)
  fields = map(trian.trians) do t
    CellField(f,t)
  end
  DistributedCellField(fields,trian)
end

function CellData.CellField(f::Number,trian::DistributedTriangulation)
  fields = map(trian.trians) do t
    CellField(f,t)
  end
  DistributedCellField(fields,trian)
end

function CellData.CellField(
  f::AbstractArray{<:AbstractArray{<:Number}},trian::DistributedTriangulation)
  fields = map(f,trian.trians) do f,t
    CellField(f,t)
  end
  DistributedCellField(fields,trian)
end

# Evaluation

function (f::DistributedCellField)(x::DistributedCellPoint)
  evaluate!(nothing,f,x)
end

function Arrays.evaluate!(cache,f::DistributedCellField,x::DistributedCellPoint)
  map(local_views(f),local_views(x)) do f,x
    evaluate!(nothing,f,x)
  end
end

# Given local CellFields and a set of original DistributedTriangulations, 
# returns the DistributedTriangulation where the local CellFields are defined. 
function _select_triangulation(fields,parents::DistributedCellField...)
  trian_candidates = unique(objectid,map(get_triangulation,parents))
  _select_triangulation(fields,trian_candidates...)
end

function _select_triangulation(fields,trian_candidates::DistributedTriangulation...)
  if length(trian_candidates) == 1
    return first(trian_candidates)
  end

  # Check if we can select one of the original triangulations
  trians = map(local_views,trian_candidates)
  t_id = map(fields,trians...) do f, trians...
    f_id = objectid(get_triangulation(f))
    return findfirst(tt -> objectid(tt) == f_id, trians)
  end |> getany
  if !isnothing(t_id)
    return trian_candidates[t_id]
  end

  # If not, check if we can build a new DistributedTriangulation based on one of the original models. 
  m_id = map(fields,trians...) do f, trians...
    f_id = objectid(get_background_model(get_triangulation(f)))
    return findfirst(tt -> objectid(get_background_model(tt)) == f_id, trians)
  end |> getany
  if !isnothing(m_id)
    model = get_background_model(trian_candidates[m_id])
    return DistributedTriangulation(map(get_triangulation,fields),model)
  end

  @error "Cannot select a triangulation for the operation"
end

# Operations

function Arrays.evaluate!(cache,k::Operation,a::DistributedCellField)
  fields = map(local_views(a)) do f
    evaluate!(nothing,k,f)
  end
  DistributedCellField(fields,get_triangulation(a))
end

function Arrays.evaluate!(
  cache,k::Operation,a::DistributedCellField,b::DistributedCellField)
  fields = map(local_views(a),local_views(b)) do f,g
    evaluate!(nothing,k,f,g)
  end
  trian = _select_triangulation(fields,a,b)
  DistributedCellField(fields,trian)
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedCellField,b::Number)
  fields = map(local_views(a)) do f
    evaluate!(nothing,k,f,b)
  end
  DistributedCellField(fields,get_triangulation(a))
end

function Arrays.evaluate!(cache,k::Operation,b::Number,a::DistributedCellField)
  fields = map(local_views(a)) do f
    evaluate!(nothing,k,b,f)
  end
  DistributedCellField(fields,get_triangulation(a))
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedCellField,b::Function)
  fields = map(local_views(a)) do f
    evaluate!(nothing,k,f,b)
  end
  DistributedCellField(fields,get_triangulation(a))
end

function Arrays.evaluate!(cache,k::Operation,b::Function,a::DistributedCellField)
  fields = map(local_views(a)) do f
    evaluate!(nothing,k,b,f)
  end
  DistributedCellField(fields,get_triangulation(a))
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedCellField...)
  fields = map(map(local_views,a)...) do f...
    evaluate!(nothing,k,f...)
  end
  trian = _select_triangulation(fields,a...)
  DistributedCellField(fields,trian)
end

# Differential ops

function Fields.gradient(a::DistributedCellField)
  DistributedCellField(map(gradient,a.fields),get_triangulation(a))
end

function Fields.divergence(a::DistributedCellField)
  DistributedCellField(map(divergence,a.fields),get_triangulation(a))
end

function Fields.DIV(a::DistributedCellField)
  DistributedCellField(map(DIV,a.fields),get_triangulation(a))
end

function Fields.∇∇(a::DistributedCellField)
  DistributedCellField(map(∇∇,a.fields),get_triangulation(a))
end

function Fields.curl(a::DistributedCellField)
  DistributedCellField(map(curl,a.fields),get_triangulation(a))
end

# Integration related
"""
"""
struct DistributedMeasure{A<:AbstractArray{<:Measure},B<:DistributedTriangulation} <: GridapType
  measures::A
  trian::B
end

local_views(a::DistributedMeasure) = a.measures

function CellData.Measure(t::DistributedTriangulation,args...;kwargs...)
  measures = map(t.trians) do trian
    Measure(trian,args...;kwargs...)
  end
  DistributedMeasure(measures,t)
end

function CellData.Measure(tt::DistributedTriangulation{Dc,Dp},it::DistributedTriangulation{Dc,Dp},args...) where {Dc,Dp}
  measures = map(local_views(tt),local_views(it)) do ttrian, itrian
    Measure(ttrian,itrian,args...)
  end
  return DistributedMeasure(measures,it)
end

function CellData.get_cell_points(a::DistributedMeasure)
  DistributedCellPoint(map(get_cell_points,a.measures),a.trian)
end

"""
"""
struct DistributedDomainContribution{A<:AbstractArray{<:DomainContribution}} <: GridapType
  contribs::A
end

CellData.num_domains(a::DistributedDomainContribution) = CellData.num_domains(getany(local_views(a)))

local_views(a::DistributedDomainContribution) = a.contribs

function Base.getindex(c::DistributedDomainContribution,t::DistributedTriangulation)
  map(getindex,c.contribs,t.trians)
end

function Fields.integrate(f::DistributedCellField,b::DistributedMeasure)
  contribs = map(f.fields,b.measures) do f,m
    integrate(f,m)
  end
  DistributedDomainContribution(contribs)
end

function Fields.integrate(f::Function,b::DistributedMeasure)
  contribs = map(b.measures) do m
    integrate(f,m)
  end
  DistributedDomainContribution(contribs)
end

function Fields.integrate(f::Number,b::DistributedMeasure)
  contribs = map(b.measures) do m
    integrate(f,m)
  end
  DistributedDomainContribution(contribs)
end

function (*)(a::Integrand,b::DistributedMeasure)
  integrate(a.object,b)
end

(*)(b::DistributedMeasure,a::Integrand) = a*b

function Base.sum(a::DistributedDomainContribution)
  sum(map(sum,a.contribs))
end

function (+)(a::DistributedDomainContribution,b::DistributedDomainContribution)
  contribs = map(+,a.contribs,b.contribs)
  DistributedDomainContribution(contribs)
end

function (-)(a::DistributedDomainContribution,b::DistributedDomainContribution)
  contribs = map(-,a.contribs,b.contribs)
  DistributedDomainContribution(contribs)
end

function (*)(a::Number,b::DistributedDomainContribution)
  contribs = map(b.contribs) do b
    a*b
  end
  DistributedDomainContribution(contribs)
end

(*)(a::DistributedDomainContribution,b::Number) = b*a

# Jordi: This is ugly, but it is useful to re-use code from Gridap: 
# A lot of the time, we create an empty DomainContribution and then add to it.
# By dispatching here, this kind of code works verbatim for GridapDistributed. 
# We could eventually replace this with an EmptyDomainContribution type.
function (+)(a::CellData.DomainContribution,b::DistributedDomainContribution)
  @assert iszero(CellData.num_domains(a))
  return b
end

# Triangulation related

function CellData.get_cell_points(a::DistributedTriangulation)
  DistributedCellPoint(map(get_cell_points,a.trians),a)
end

function CellData.get_normal_vector(a::DistributedTriangulation)
  fields = map(get_normal_vector,a.trians)
  DistributedCellField(fields,a)
end

# Skeleton related

function DistributedCellField(a::AbstractArray{<:SkeletonPair},trian::DistributedTriangulation)
  plus, minus = map(s->(s.plus,s.minus),a) |> tuple_of_arrays
  tplus  = _select_triangulation(plus,trian)
  tminus = _select_triangulation(minus,trian)
  dplus  = DistributedCellField(plus,tplus)
  dminus = DistributedCellField(minus,tminus)
  SkeletonPair(dplus,dminus)
end

function Base.getproperty(x::DistributedCellField, sym::Symbol)
  if sym in (:⁺,:plus)
    DistributedCellField(map(i->i.plus,local_views(x)),get_triangulation(x))
  elseif sym in (:⁻, :minus)
    DistributedCellField(map(i->i.minus,local_views(x)),get_triangulation(x))
  else
    getfield(x, sym)
  end
end

CellData.jump(a::DistributedCellField) = DistributedCellField(map(jump,a.fields),get_triangulation(a))
CellData.mean(a::DistributedCellField) = DistributedCellField(map(mean,a.fields),get_triangulation(a))

# DistributedCellDof

struct DistributedCellDof{A<:AbstractArray{<:CellDof},B<:DistributedTriangulation} <: CellDatum
  dofs::A
  trian::B
end

local_views(s::DistributedCellDof) = s.dofs
CellData.get_triangulation(s::DistributedCellDof) = s.trian

function CellData.DomainStyle(::Type{<:DistributedCellDof{A}}) where A 
  DomainStyle(eltype(A))
end

(a::DistributedCellDof)(f) = evaluate(a,f)

function Gridap.Arrays.evaluate!(cache,s::DistributedCellDof,f::DistributedCellField)
  map(local_views(s),local_views(f)) do s, f
    evaluate!(nothing,s,f)
  end
end

function Gridap.Arrays.evaluate!(cache, ::DistributedCellField, ::DistributedCellDof)
@unreachable """\n
CellField (f) objects cannot be evaluated at CellDof (s) objects.
However, CellDofs objects can be evaluated at CellField objects.
Did you mean evaluate(f,s) instead of evaluate(s,f), i.e.
f(s) instead of s(f)?
"""
end

# Interpolation at arbitrary points (returns -Inf if the point is not found)
Arrays.evaluate!(cache,f::DistributedCellField,x::Point) = evaluate(Interpolable(f),x)
Arrays.evaluate!(cache,f::DistributedCellField,x::AbstractVector{<:Point}) = evaluate(Interpolable(f),x)

struct DistributedInterpolable{Tx,Ty,A} <: Function
  interps::A
  function DistributedInterpolable(interps::AbstractArray{<:Interpolable})
    Tx,Ty = map(interps) do I
      trian = get_triangulation(I.uh)
      x = mean(testitem(get_cell_coordinates(trian)))
      return typeof(x), return_type(I,x)
    end |> tuple_of_arrays
    Tx = getany(Tx)
    Ty = getany(Ty)
    A  = typeof(interps)
    new{Tx,Ty,A}(interps)
  end
end

local_views(a::DistributedInterpolable) = a.interps

function Interpolable(f::DistributedCellField;kwargs...) 
  interps = map(local_views(f)) do f
    Interpolable(f,kwargs...)
  end
  DistributedInterpolable(interps)
end

(a::DistributedInterpolable)(x) = evaluate(a,x)

Arrays.return_cache(f::DistributedInterpolable,x::Point) = return_cache(f,[x])
Arrays.evaluate!(caches,I::DistributedInterpolable,x::Point) = first(evaluate!(caches,I,[x]))

function Arrays.return_cache(I::DistributedInterpolable{Tx,Ty},x::AbstractVector{<:Point}) where {Tx,Ty}
  msg = "Can only evaluate DistributedInterpolable at physical points of the same dimension of the underlying triangulation"
  @check Tx == eltype(x) msg
  caches = map(local_views(I)) do I
    trian = get_triangulation(I.uh)
    y = mean(testitem(get_cell_coordinates(trian)))
    return_cache(I,y)
  end
  caches
end

function Arrays.evaluate!(cache,I::DistributedInterpolable{Tx,Ty},x::AbstractVector{<:Point}) where {Tx,Ty}
  _allgather(x) = PartitionedArrays.getdata(getany(gather(x;destination=:all)))

  # Evaluate in local portions of the domain. Only keep points inside the domain.
  nx = length(x)
  my_ids, my_vals = map(local_views(I),local_views(cache)) do I, cache
    ids  = Vector{Int}(undef,nx)
    vals = Vector{Ty}(undef,nx)
    k = 1
    yi = zero(Ty)
    for (i,xi) in enumerate(x)
      inside = true
      try
        yi = evaluate!(cache,I,xi)
      catch
        inside = false
      end
      if inside
        ids[k] = i
        vals[k] = copy(yi)
        k += 1
      end
    end
    resize!(ids,k-1)
    resize!(vals,k-1)
    return ids, vals
  end |> tuple_of_arrays

  # Communicate results, so that every (id,value) pair is known by every process
  if Ty <: VectorValue
    D = num_components(Ty)
    vals_d = Vector{Vector{eltype(Ty)}}(undef,D)
    for d in 1:D
      my_vals_d = map(y_p -> map(y_p_i -> y_p_i[d],y_p),my_vals)
      vals_d[d] = _allgather(my_vals_d)
    end
    vals = map(VectorValue,vals_d...)
  else
    vals = _allgather(my_vals)
  end
  ids = _allgather(my_ids)

  # Combine results
  w = Vector{Ty}(undef,nx)
  for (i,v) in zip(ids,vals)
    w[i] = v
  end

  return w
end

# Support for distributed Dirac deltas
struct DistributedDiracDelta{D} <: GridapType
  Γ::DistributedTriangulation
  dΓ::DistributedMeasure
end
# This code is from Gridap, repeated in BoundaryTriangulations.jl and DiracDelta.jl.
# We could refactor...
import Gridap.Geometry: FaceToCellGlue
function BoundaryTriangulation{D}(
  model::DiscreteModel,
  face_to_bgface::AbstractVector{<:Integer}) where D

  bgface_to_lcell = Fill(1,num_faces(model,D))

  topo = get_grid_topology(model)
  bgface_grid = Grid(ReferenceFE{D},model)
  face_grid = view(bgface_grid,face_to_bgface)
  cell_grid = get_grid(model)
  glue = FaceToCellGlue(topo,cell_grid,face_grid,face_to_bgface,bgface_to_lcell)
  trian = BodyFittedTriangulation(model,face_grid,face_to_bgface)
  BoundaryTriangulation(trian,glue)
end

function BoundaryTriangulation{D}(model::DiscreteModel;tags) where D
  labeling = get_face_labeling(model)
  bgface_to_mask = get_face_mask(labeling,tags,D)
  face_to_bgface = findall(bgface_to_mask)
  BoundaryTriangulation{D}(model,face_to_bgface)
end

function DiracDelta{D}(model::DistributedDiscreteModel{Dc},degree::Integer;kwargs...) where {D,Dc}

  @assert 0 <= D && D < num_cell_dims(model) """\n
  Incorrect value of D=$D for building a DiracDelta{D} on a model with $(num_cell_dims(model)) cell dims.

  D should be in [0,$(num_cell_dims(model))).
  """

  gids   = get_face_gids(model,Dc)
  trians = map(local_views(model),partition(gids)) do model, gids
      trian = BoundaryTriangulation{D}(model;kwargs...)
      filter_cells_when_needed(no_ghost,gids,trian)
  end
  Γ=DistributedTriangulation(trians,model)
  dΓ=Measure(Γ,degree)
  DistributedDiracDelta{D}(Γ,dΓ)
end

# Following functions can be eliminated introducing an abstract delta in Gridap.jl
function DiracDelta{0}(model::DistributedDiscreteModel;tags)
  degree = 0
  DiracDelta{0}(model,degree;tags=tags)
end
function (d::DistributedDiracDelta)(f)
 evaluate(d,f)
end
function Gridap.Arrays.evaluate!(cache,d::DistributedDiracDelta,f)
 ∫(f)*d.dΓ
end
