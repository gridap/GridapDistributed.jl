
# Distributed Constraints

This is a framework for the implementation of distributed constraints. To reduce the number of communications, we rely on one core invariant:

> If a slave DoF is **owned** by a process, all its master DoFs are **local** to that process (owned or ghost).

Hanging node constraints, periodicity constraints, and AgFEM constraints can all be implemented within this framework, **provided the local triangulation is built with enough overlap** that this invariant holds. Concretely:

- **Hanging nodes**: masters are vertices/edges of the across-the-face neighbor cell, which is already inside the standard 1-cell ghost layer.
- **AgFEM**: the local triangulation is built from owned agglomerates **plus one extra layer of agglomerates**. Since AgFEM background meshes are typically Cartesian, this extra overlap is cheap.
- **Periodicity**: the local triangulation is extended by one layer on each periodic boundary *at construction time*, and periodic partner nodes are identified through this overlap (e.g. for a 4-segment line split across 2 processors, extend each side by one segment and identify the first node of P1 with the second-to-last node of P2, and vice versa). This requires periodicity information to be known **before partitioning**, which is a real but acceptable cost: it is paid once at mesh-construction time, not as a runtime communication pattern.

With sufficient overlap, every owned slave's masters — including masters-of-masters in a chain — are already present in the local triangulation. Chains are therefore resolved **locally**, without any extra communication.

The two methods the (distributed) `ConstraintHandler` needs to support are the same as in serial:

- `merge!`: merge several local sets of constraints `A, B, C, ...` into one.
- `close!`: close the DAG so that no DoF is both a master and a slave.

In addition we use `consistent!`, with the same meaning as in `PartitionedArrays`: it synchronizes the constraint definition of a DoF from its owner to its ghost copies. There is no need for `assemble!` — constraints are never accumulated across processes, only copied.

## Setting up and merging local constraints

Each process sets up its own local constraints `Ak`, `Bk`, `Ck`, ... for owned slave DoFs. Because of the invariant above (with sufficient overlap), this requires **no communication**: all masters needed are already local.

`merge!(Ak, Bk, Ck, ...)` combines these into a single local set, also with **no communication**.

Constraints that this process happens to set up for *non-owned* (ghost) slave DoFs may be incomplete or outdated — they will be overwritten by `consistent!` below, so their presence is harmless.

## Making constraints consistent

After local merging, each process holds a set of constraints whose **owned slaves are correct**, but whose ghost slaves may not be.

`consistent!` performs **one round of nearest-neighbor communication**: each owner sends the constraint definition (masters + coefficients, by **global** DoF id) of its owned slaves to all processes holding that DoF as a ghost.

On the receiving side, incoming master DoFs are mapped from global to local ids. If a master's global id does not correspond to any DoF currently in the local index space, a new **fictitious local id** is created for it. This is why the local constraint space may contain DoFs that do not belong to the original local FE space — they exist purely as masters referenced by constraints on local slaves.

After this single round, every process sees a complete, consistent local view of the constraint DAG restricted to its local DoFs (owned + ghost + fictitious).

## Closing constraints locally

`close!` can now be run **independently and locally on each process**, with no further communication. Recall that `close!`'s chain-resolution step is an **iterative substitution to a fixed point** (not a single topological pass): it repeatedly replaces any master that is itself a slave with that master's own masters, until no constrained DoF appears as a master anywhere. For a DAG, this fixed point is uniquely determined by the constraint graph itself — it does **not** depend on:

- the order constraints are stored/iterated in,
- the order masters appear within a constraint line,
- which substitutions happen to occur in which pass.

So as long as every process's local DAG (after `consistent!`) agrees on the same global DoFs and the same constraint graph, `close!` produces identical results on every process. **No global ordering or extra communication is needed for this step.**

## Canonical ordering for tie-breaking

The one place where processes *do* need to agree is **discrete resolution choices** — situations where the algorithm must pick one of several otherwise-equivalent options, and all processes must pick the *same* one.

The clearest example is resolving an over-determined-but-consistent cycle (e.g. a periodic corner where `A = B`, `B = C`, `C = A` all with coefficient 1): the resolution is to pick one DoF as the canonical master and rewrite the others in terms of it. If two processes pick different DoFs as the master for the same cycle, the resulting constraint sets are mutually inconsistent.

This is solved without communication: any such tie-break must be made using a **canonical, globally shared key** — the **global DoF id** — rather than local array order or local index. E.g. "pick the DoF with the smallest global id as master." Since global ids are already shared (no extra communication), every process makes the same choice given the same set of candidates.

**Rule of thumb:** anywhere the algorithm needs to break a tie or pick among equivalent options, sort/compare by global DoF id, never by local index or array position.

## Relaxing the invariant: per-source overlap + index set union

The core invariant ("owned slave ⇒ local masters") was originally framed as a single global property the local triangulation must satisfy. In practice it is more useful — and easier to satisfy — as a **per-constraint-source** property, reconciled afterwards by a purely local bookkeeping step.

**Each constraint source builds its own index set.** Each source produces a constraint set `A`, `B`, `C`, ... together with **its own partitioned index set**, recording the global ids of any "extra" (fictitious) DoFs it referenced as masters.

**Unifying the index sets is local.** Before merging `A`, `B`, `C`, ...:

1. Take the **union** of their index sets (a set of global ids) — purely local, no communication. This defines a single, process-local numbering covering every DoF (owned, ghost, or fictitious) referenced by any of the sets.

2. **Reindex** each set's constraints into this unified numbering. Only the fictitious masters need remapping — owned/ghost DoFs already share a common local id across all sets.

After reindexing, every master referenced by any set has a valid local id in the unified space — i.e. the invariant now holds *for the merged set*, even though no single set satisfied it w.r.t. a common index space beforehand. `merge!`, `close!`, and the final `consistent!` then proceed exactly as described above.

This works without extra communication because of a simple fact: `merge!` never introduces a master that wasn't already a master in one of the input sets. So if each source's index set already covers every master *that source* refers to (a per-source requirement, no harder than the single-source case), the union automatically covers every master in the merged set — including cross-source chains (e.g. a slave in `A`'s overlap that turns out to be defined as a slave by `B`: its master, in turn, is necessarily in `B`'s index set, hence in the union).

**Free vs. slave-valued masters.** The invariant only needs to hold for masters that are themselves **slaves** — those are the ones whose *definition* must be locally available for `close!`'s substitution to work. A master that is **free** (unconstrained) never needs a local id at all for `close!`'s purposes: it can remain a bare global-id reference, à la deal.II, resolved only at assembly/`distribute(x)` time like any other off-process DoF. This shrinks the overlap each source actually needs to provide.
