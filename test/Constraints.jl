
using Gridap
using Gridap.Arrays: Table
using Gridap.Geometry: get_node_coordinates
using GridapDistributed
using PartitionedArrays
using Test

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

const TOL = 1e-10

# Local free-dof index of the node at (x,y), or nothing.
# For Q1 with no Dirichlet BCs: local_dof == local_node.
function dof_at(sp, x, y)
  coords = get_node_coordinates(get_triangulation(sp))
  for (i, c) in enumerate(coords)
    abs(c[1] - x) < TOL && abs(c[2] - y) < TOL && return i
  end
  return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
#
# 4×2 Cartesian mesh on [0,1]², partition (2,1). No Dirichlet BCs → 15 free DOFs.
#
# Global node layout:
#  y=1:   5(0,1)   6(¼,1)  13(½,1)  14(¾,1)  15(1,1)
#  y=½:   3(0,½)   4(¼,½)  10(½,½)  11(¾,½)  12(1,½)
#  y=0:   1(0,0)   2(¼,0)   7(½,0)   8(¾,0)   9(1,0)
#
# Ownership:
#  P1 owns x∈{0,¼}   → globals 1,2,3,4,5,6;    ghosts x∈{½,¾} → 7,8,10,11,13,14
#  P2 owns x∈{½,¾,1} → globals 7,8,…,15;        ghosts x=¼      → 2,4,6
#
# Constraint scenarios:
#
#  [S1] Owned slave on P1, all masters owned by P1:
#         (¼,½) → 0.5·(¼,0) + 0.5·(¼,1)
#
#  [S2] Ghost slave on P1 (P2-owned), master IS in P1's local index (ghost):
#         (¾,0) → 1.0·(½,0)
#
#  [S3] Ghost slave on P1 (P2-owned), one master NOT in P1's local index (fictitious):
#         (¾,½) → 0.5·(½,½) + 0.5·(1,½)

np        = (2, 1)
ranks     = collect(LinearIndices((prod(np),)))
model     = CartesianDiscreteModel(ranks, np, (0,1,0,1), (4,2))
V         = FESpace(model, ReferenceFE(lagrangian, Float64, 1))
cell_gids = GridapDistributed.get_cell_gids(model)
spaces    = local_views(V)
dof_ids   = partition(get_free_dof_ids(V))

# ──────────────────────────────────────────────────────────────────────────────
# Build per-part constraint tables
# ──────────────────────────────────────────────────────────────────────────────
#
# Owned slaves supply real master local-dof indices + coefficients.
# Ghost slaves send empty rows; consistent! overwrites them from the owner.
# Rows are sorted ascending by local-dof index (required by generate_constraint_gids).

owned_constraints = Dict(
  (0.25, 0.5) => ([(0.25, 0.0), (0.25, 1.0)], [0.5, 0.5]),  # S1: owned by P1
  (0.75, 0.0) => ([(0.50, 0.0)],               [1.0]      ),  # S2: owned by P2
  (0.75, 0.5) => ([(0.50, 0.5), (1.00, 0.5)],  [0.5, 0.5]),  # S3: owned by P2
)

_sDOF_to_dof_parts, sDOF_to_mdofs_parts, sDOF_to_coeffs_parts = map(
  spaces, dof_ids
) do sp, ids
  l2own  = local_to_own(ids)
  entries = Tuple{Int32, Vector{Int32}, Vector{Float64}}[]

  for ((sx, sy), (mcoords, cs)) in owned_constraints
    ld = dof_at(sp, sx, sy)
    ld === nothing && continue

    if !iszero(l2own[ld])        # owned slave: fill real masters
      mdofs = Int32[dof_at(sp, mc...) for mc in mcoords]
      @assert all(!isnothing, mdofs)
      push!(entries, (Int32(ld), mdofs, Float64.(cs)))
    else                         # ghost slave: pre-size row to match owner; consistent! fills values
      n = length(mcoords)
      push!(entries, (Int32(ld), zeros(Int32, n), zeros(Float64, n)))
    end
  end

  sort!(entries; by = first)     # ascending local-dof order

  sdofs = Int32[e[1] for e in entries]
  ptrs  = Int32[1; 1 .+ cumsum(length(e[2]) for e in entries)]
  mdata = reduce(vcat, getindex.(entries, 2); init = Int32[])
  cdata = reduce(vcat, getindex.(entries, 3); init = Float64[])
  sdofs, Table(mdata, ptrs), Table(cdata, ptrs)
end |> tuple_of_arrays

# ──────────────────────────────────────────────────────────────────────────────
# Call generate_distributed_constraints (single-constraint path)
# ──────────────────────────────────────────────────────────────────────────────

sDOF_gids, new_mfdof_gids, new_mddof_gids,
mDOF_to_dof, sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs =
  GridapDistributed.generate_distributed_constraints(
    cell_gids, spaces,
    _sDOF_to_dof_parts, sDOF_to_mdofs_parts, sDOF_to_coeffs_parts
  )

# ──────────────────────────────────────────────────────────────────────────────
# Helpers for per-part queries
# ──────────────────────────────────────────────────────────────────────────────

# Return (coeffs, master_global_ids) for the slave at (sx,sy) on one part, or nothing.
# sDOF_to_mdofs entries are signed mDOF local indices: positive → mfdof, negative → mddof.
function query_slave(sp, mfgids, sdof_vec, smdofs, scoeffs, sx, sy)
  ld  = dof_at(sp, sx, sy)
  ld  === nothing && return nothing
  idx = findfirst(==(Int32(ld)), sdof_vec)
  idx === nothing && return nothing
  coeffs   = scoeffs.data[scoeffs.ptrs[idx]:scoeffs.ptrs[idx+1]-1]
  mrow     = smdofs.data[smdofs.ptrs[idx]:smdofs.ptrs[idx+1]-1]
  l2g_mf   = local_to_global(mfgids)
  mgids    = [e > 0 ? l2g_mf[e] : error("unexpected mddof entry") for e in mrow]
  return coeffs, mgids
end

# Unpack distributed outputs into named per-part tuples for easy indexing
parts = map(
  spaces, dof_ids,
  partition(sDOF_gids), partition(new_mfdof_gids),
  sDOF_to_dof, sDOF_to_mdofs, sDOF_to_coeffs
) do sp, ids, sgids, mfgids, sdof_vec, smdofs, scoeffs
  (; sp, ids, sgids, mfgids, sdof_vec, smdofs, scoeffs)
end
p1, p2 = parts[1], parts[2]

# ──────────────────────────────────────────────────────────────────────────────
# TEST 1 – Slave global ids are consistent across processes
# ──────────────────────────────────────────────────────────────────────────────
# (¾,0) is owned by P2 and a ghost on P1; both must agree on its global slave id.

@testset "Slave gids are consistent" begin
  gid_p1 = let d = dof_at(p1.sp, 0.75, 0.0),
                i = findfirst(==(Int32(d)), p1.sdof_vec)
    local_to_global(p1.sgids)[i]
  end
  gid_p2 = let d = dof_at(p2.sp, 0.75, 0.0),
                i = findfirst(==(Int32(d)), p2.sdof_vec)
    local_to_global(p2.sgids)[i]
  end
  @test gid_p1 == gid_p2
end

# ──────────────────────────────────────────────────────────────────────────────
# TEST 2 – [S1] Owned slave (¼,½) on P1: coefficients and master global ids
# ──────────────────────────────────────────────────────────────────────────────

@testset "S1: owned slave, local masters (P1)" begin
  r = query_slave(p1.sp, p1.mfgids, p1.sdof_vec, p1.smdofs, p1.scoeffs, 0.25, 0.5)
  @test r !== nothing
  coeffs, mgids = r
  @test coeffs ≈ [0.5, 0.5]

  g_m1 = local_to_global(p1.ids)[dof_at(p1.sp, 0.25, 0.0)]
  g_m2 = local_to_global(p1.ids)[dof_at(p1.sp, 0.25, 1.0)]
  @test Set(mgids) == Set([g_m1, g_m2])
end

# ──────────────────────────────────────────────────────────────────────────────
# TEST 3 – [S1] Ghost copy of (¼,½) on P2 receives P1's constraint via consistent!
# ──────────────────────────────────────────────────────────────────────────────
# After consistent!, P2's ghost row for (¼,½) should carry the same coefficients
# and master global ids. Masters (¼,0) and (¼,1) are ghosts on P2.

@testset "S1: ghost copy on P2 after consistent!" begin
  r = query_slave(p2.sp, p2.mfgids, p2.sdof_vec, p2.smdofs, p2.scoeffs, 0.25, 0.5)
  @test r !== nothing
  coeffs, mgids = r
  @test coeffs ≈ [0.5, 0.5]

  g_m1 = local_to_global(p2.ids)[dof_at(p2.sp, 0.25, 0.0)]
  g_m2 = local_to_global(p2.ids)[dof_at(p2.sp, 0.25, 1.0)]
  @test Set(mgids) == Set([g_m1, g_m2])
end

# ──────────────────────────────────────────────────────────────────────────────
# TEST 4 – [S2] Ghost slave (¾,0) on P1: consistent! delivers P2's constraint;
#          master (½,0) is a ghost on P1 (in local index)
# ──────────────────────────────────────────────────────────────────────────────

@testset "S2: ghost slave, master local on P1 after consistent!" begin
  # P1 view: ghost slave receives constraint from P2
  r1 = query_slave(p1.sp, p1.mfgids, p1.sdof_vec, p1.smdofs, p1.scoeffs, 0.75, 0.0)
  @test r1 !== nothing
  coeffs1, mgids1 = r1
  @test coeffs1 ≈ [1.0]
  g_half_0 = local_to_global(p1.ids)[dof_at(p1.sp, 0.5, 0.0)]   # ghost on P1
  @test mgids1 == [g_half_0]

  # P2 view: owned slave, master (½,0) is owned by P2
  r2 = query_slave(p2.sp, p2.mfgids, p2.sdof_vec, p2.smdofs, p2.scoeffs, 0.75, 0.0)
  @test r2 !== nothing
  coeffs2, mgids2 = r2
  @test coeffs2 ≈ [1.0]
  g_half_0_p2 = local_to_global(p2.ids)[dof_at(p2.sp, 0.5, 0.0)]  # owned by P2
  @test mgids2 == [g_half_0_p2]

  # Both sides must agree on the master's global id
  @test g_half_0 == g_half_0_p2
end

# ──────────────────────────────────────────────────────────────────────────────
# TEST 5 – [S3] Ghost slave (¾,½) on P1: one master (1,½) not in P1's local
#          index → must appear as a fictitious dof in new_mfdof_gids
# ──────────────────────────────────────────────────────────────────────────────

@testset "S3: ghost slave, non-local master → fictitious dof on P1" begin
  # (1,½) is owned by P2; its global id must appear in P1's extended mfdof partition
  gid_1_half = local_to_global(p2.ids)[dof_at(p2.sp, 1.0, 0.5)]
  @test gid_1_half ∈ local_to_global(p1.mfgids)

  # P1 ghost slave (¾,½) must carry correct coefficients and master global ids
  r = query_slave(p1.sp, p1.mfgids, p1.sdof_vec, p1.smdofs, p1.scoeffs, 0.75, 0.5)
  @test r !== nothing
  coeffs, mgids = r
  @test coeffs ≈ [0.5, 0.5]

  gid_half_half = local_to_global(p1.ids)[dof_at(p1.sp, 0.5, 0.5)]  # ghost on P1
  @test Set(mgids) == Set([gid_half_half, gid_1_half])

  # P2 owned slave sanity check: same coefficients
  r2 = query_slave(p2.sp, p2.mfgids, p2.sdof_vec, p2.smdofs, p2.scoeffs, 0.75, 0.5)
  @test r2 !== nothing
  @test r2[1] ≈ [0.5, 0.5]
end
