
using Gridap
using Gridap.Arrays: Table
using Gridap.Geometry: get_node_coordinates
using GridapDistributed
using PartitionedArrays
using Test

# 4×2 Cartesian mesh on [0,1]², 2-part partition (left/right). No Dirichlet BCs.
#
# Node layout (global IDs):
#   y=1:  5(0,1)   6(¼,1)  13(½,1)  14(¾,1)  15(1,1)
#   y=½:  3(0,½)   4(¼,½)  10(½,½)  11(¾,½)  12(1,½)
#   y=0:  1(0,0)   2(¼,0)   7(½,0)   8(¾,0)   9(1,0)
#
# P1 owns x∈{0,¼}, P2 owns x∈{½,¾,1}.
#
# Three constraints stressing all cases:
#   S1: (¼,½) = 0.5·(¼,0) + 0.5·(¼,1)  — owned by P1, masters local on both parts
#   S2: (¾,0) = 1.0·(½,0)               — owned by P2, master local on both parts (ghost on P1)
#   S3: (¾,½) = 0.5·(½,½) + 0.5·(1,½)  — owned by P2, (1,½) is fictitious on P1

np    = (2, 1)
ranks = DebugArray(LinearIndices((prod(np),)))
model = CartesianDiscreteModel(ranks, np, (0,1,0,1), (4,2))
V     = FESpace(model, ReferenceFE(lagrangian, Float64, 1))
cell_gids = GridapDistributed.get_cell_gids(model)
spaces    = local_views(V)
dof_ids   = partition(get_free_dof_ids(V))

# Local DOF index of the node at (x,y), or nothing if not in this part's local index.
function dof_at(sp, x, y; tol=1e-10)
  for (i, c) in enumerate(get_node_coordinates(get_triangulation(sp)))
    abs(c[1]-x) < tol && abs(c[2]-y) < tol && return i
  end
  return nothing
end

# Owned rows carry real master local dofs; ghost rows are pre-sized with zeros
# (consistent! fills them in from the owner).
owned_constraints = Dict(
  (0.25, 0.5) => ([(0.25, 0.0), (0.25, 1.0)], [0.5, 0.5]),  # S1, owned by P1
  (0.75, 0.0) => ([(0.50, 0.0)],               [1.0]      ),  # S2, owned by P2
  (0.75, 0.5) => ([(0.50, 0.5), (1.00, 0.5)],  [0.5, 0.5]),  # S3, owned by P2
)

_sDOF_to_dof_parts, sDOF_to_mdofs_parts, sDOF_to_coeffs_parts = map(
  spaces, dof_ids
) do sp, ids
  l2own = local_to_own(ids)
  entries = Tuple{Int32, Vector{Int32}, Vector{Float64}}[]
  for ((sx, sy), (mcoords, cs)) in owned_constraints
    ld = dof_at(sp, sx, sy)
    ld === nothing && continue
    if !iszero(l2own[ld])
      mdofs = Int32[dof_at(sp, mc...) for mc in mcoords]
      push!(entries, (Int32(ld), mdofs, Float64.(cs)))
    else
      push!(entries, (Int32(ld), zeros(Int32, length(mcoords)), zeros(Float64, length(mcoords))))
    end
  end
  sort!(entries; by = first)
  sdofs = Int32[e[1] for e in entries]
  ptrs  = Int32[1; 1 .+ cumsum(length(e[2]) for e in entries)]
  mdata = reduce(vcat, getindex.(entries, 2); init = Int32[])
  cdata = reduce(vcat, getindex.(entries, 3); init = Float64[])
  sdofs, Table(mdata, ptrs), Table(cdata, ptrs)
end |> tuple_of_arrays

sDOF_gids, new_mfdof_gids, new_mddof_gids,
mDOF_to_dof, sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs =
  GridapDistributed.generate_distributed_constraints(
    cell_gids, spaces,
    _sDOF_to_dof_parts, sDOF_to_mdofs_parts, sDOF_to_coeffs_parts
  )

# For the slave at (sx,sy), return sorted [(master_local_dof, coeff)] pairs.
# master_local_dof = mDOF_to_dof[m] for an mfdof entry m > 0;
# entries with mDOF_to_dof = 0 are fictitious (master not in this part's local index).
function slave_data(sp, sdof_vec, smdofs, scoeffs, mdof_to_dof, sx, sy)
  ld  = dof_at(sp, sx, sy)
  ld  === nothing && return nothing
  idx = findfirst(==(Int32(ld)), sdof_vec)
  idx === nothing && return nothing
  mrow   = smdofs.data[smdofs.ptrs[idx]:smdofs.ptrs[idx+1]-1]
  coeffs = scoeffs.data[scoeffs.ptrs[idx]:scoeffs.ptrs[idx+1]-1]
  mdofs  = [m > 0 ? Int(mdof_to_dof[m]) : error("unexpected mddof") for m in mrow]
  return sort!(collect(zip(mdofs, coeffs)))
end

@testset "generate_distributed_constraints" begin
  map(spaces, sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs, mDOF_to_dof) do sp, sdof_vec, smdofs, scoeffs, mdof_to_dof

    # S1: (¼,½) = 0.5·(¼,0) + 0.5·(¼,1)
    # Slave owned by P1, ghost on P2. Masters are local on both parts.
    if dof_at(sp, 0.25, 0.5) !== nothing
      d_m1 = dof_at(sp, 0.25, 0.0)
      d_m2 = dof_at(sp, 0.25, 1.0)
      @test slave_data(sp, sdof_vec, smdofs, scoeffs, mdof_to_dof, 0.25, 0.5) ==
            sort([(d_m1, 0.5), (d_m2, 0.5)])
    end

    # S2: (¾,0) = 1.0·(½,0)
    # Slave owned by P2, ghost on P1. Master (½,0) is local on both parts.
    if dof_at(sp, 0.75, 0.0) !== nothing
      d_m = dof_at(sp, 0.5, 0.0)
      @test slave_data(sp, sdof_vec, smdofs, scoeffs, mdof_to_dof, 0.75, 0.0) ==
            [(d_m, 1.0)]
    end

    # S3: (¾,½) = 0.5·(½,½) + 0.5·(1,½)
    # Slave owned by P2, ghost on P1. On P1, (1,½) is fictitious: mDOF_to_dof = 0.
    # On P2, both masters are local: mDOF_to_dof gives their local dof.
    if dof_at(sp, 0.75, 0.5) !== nothing
      d_m1 = dof_at(sp, 0.5, 0.5)
      d_m2 = coalesce(dof_at(sp, 1.0, 0.5), 0)  # 0 when fictitious on P1
      @test slave_data(sp, sdof_vec, smdofs, scoeffs, mdof_to_dof, 0.75, 0.5) ==
            sort([(d_m1, 0.5), (d_m2, 0.5)])
    end

  end
end
