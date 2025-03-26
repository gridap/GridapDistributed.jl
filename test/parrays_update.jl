
using Gridap
using PartitionedArrays
using GridapDistributed

using Gridap.FESpaces, Gridap.Algebra

np = (1,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

nc = (2,4)
serial_model = CartesianDiscreteModel((0,1,0,1),nc)
dist_model = CartesianDiscreteModel(ranks,np,(0,1,0,1),nc)

cids = get_cell_gids(dist_model)

reffe = ReferenceFE(lagrangian,Float64,1)
serial_V = TestFESpace(serial_model,reffe)
dist_V = TestFESpace(dist_model,reffe)

serial_ids = get_free_dof_ids(serial_V)
dist_ids = get_free_dof_ids(dist_V)

dof_ids = get_free_dof_ids(dist_V)

serial_Ω = Triangulation(serial_model)
serial_dΩ = Measure(serial_Ω,2)

dist_Ω = Triangulation(dist_model)
dist_dΩ = Measure(dist_Ω,2)

dist_Ωg = Triangulation(GridapDistributed.with_ghost,dist_model)
dist_dΩg = Measure(dist_Ωg,2)

serial_a(u,v) = ∫(u⋅v)*serial_dΩ
dist_a(u,v) = ∫(u⋅v)*dist_dΩ
dist_ag(u,v) = ∫(u⋅v)*dist_dΩg

serial_A = assemble_matrix(serial_a,serial_V,serial_V)

assem = SparseMatrixAssembler(dist_V,dist_V,GridapDistributed.Assembled())
dist_A_AS = assemble_matrix(dist_a,assem,dist_V,dist_V)

assem = SparseMatrixAssembler(dist_V,dist_V,GridapDistributed.LocallyAssembled())
dist_A_LA = assemble_matrix(dist_ag,assem,dist_V,dist_V)

assem = SparseMatrixAssembler(dist_V,dist_V,GridapDistributed.SubAssembled())
dist_A_SA = assemble_matrix(dist_a,assem,dist_V,dist_V)

all(centralize(dist_A_AS) - serial_A .< 1e-10)

x_AS = prand(partition(axes(dist_A_AS,2)))
x_LA = GridapDistributed.change_ghost(x_AS,axes(dist_A_LA,2))
x_SA = GridapDistributed.change_ghost(x_AS,axes(dist_A_SA,2))

norm(dist_A_AS*x_AS - dist_A_LA*x_LA)
norm(dist_A_AS*x_AS - dist_A_SA*x_SA)

assemble_matrix!(dist_a,dist_A_AS,assem,dist_V,dist_V)

############################################################################################

A = deepcopy(dist_A_AS)

t = assemble!(A;reuse=true)
_, cache = fetch(t)

t2 = assemble!(A,cache)
wait(t2)

map(==, partition(A), partition(dist_A_AS))

############################################################################################

function PartitionedArrays.precompute_nzindex(A,I,J;skip=false)
  K = zeros(Int32,length(I))
  for (p,(i,j)) in enumerate(zip(I,J))
      if !skip && (i < 1 || j < 1)
          continue
      end
      K[p] = nzindex(A,i,j)
  end
  K
end

############################################################################################

ids = cids
indices = partition(ids)

n_own = own_length(cids)
ghost2global = ghost_to_global(cids)
ghost2owner = ghost_to_owner(cids)

first_gid = scan(+,n_own,type=:exclusive,init=one(eltype(n_own)))
n_global = reduce(+,n_own)
n_parts = length(ranks)
indices2 = map(ranks,n_own,first_gid,ghost2global,ghost2owner) do rank,n_own,start,ghost2global,ghost2owner
  p = CartesianIndex((rank,))
  np = (n_parts,)
  n = (n_global,)
  ranges = ((1:n_own).+(start-1),)
  ghost = GhostIndices(n_global,ghost2global,ghost2owner)

  PartitionedArrays.LocalIndicesWithVariableBlockSize(p,np,n,ranges,ghost)
end

map(indices,indices2) do indices, indices2
  ghost2local = ghost_to_local(indices)
  own2local = own_to_local(indices)

  n_own = own_length(indices)
  n_ghost = ghost_length(indices)

  #perm = vcat(own2local,ghost2local)
  perm = fill(0,local_length(indices))
  perm[own2local] .= 1:n_own
  perm[ghost2local] .= (n_own+1):(n_ghost+n_own)
  permute_indices(indices2,perm)
end
