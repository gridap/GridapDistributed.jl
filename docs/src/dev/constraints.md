
# Distributed Constraints

This is a framework for the implementation of distributed constraints. To reduce the number of communications, we rely on one core assumption:

> If a slave DoF is **owned** by a process, all its master DoFs are **local** to that process (owned or ghost).

We later will discuss what steps require this invariant, and how to get around it if it cannot be satisfied.

Hanging node constraints, periodicity constraints, and AgFEM constraints can all be implemented within this framework (provided the local triangulation is correctly built, see the AgFEM note for more details).

We will now describe the algorithm at a high-level for both the single-constraint-source and multiple-constraint-source cases, then describe each shared step in more detail.

## Case 1: single constraint source

This is the conceptually simple case: Our assumption implies that each process can fully resolve its owned constraints. Thus, we can proceed as follows:

- Provided a globally consistent `DOF_to_is_slave` array, we build (in a single communication step) three global communicators: A slave global numbering (sDOF_gids), a free master global numbering (mfdof_gids), and a Dirichlet master global numbering (mddof_gids).
- Given sDOF_gids, we call a user-provided callback that allocated all LOCAL constraints and fills up all OWNED constraints. This requires no communication, since all masters of owned slaves are local by assumption.
- At this point, we have correct OWNED constraints. When then proceed to communicate the GHOST constraints from their owners to the processes that hold them as ghosts (one round of nearest-neighbor communication, similar to consistent!).
- After this, we have a complete set of globally consistent constraints in DOF local numbering. We can then reindex them to mdof numbering (which is what the space constructor expects).

## Case 2: multiple constraint sources

This case is a bit more complicated. The key concept is the following:

- Because our assumption, we can merge the OWNED constraints locally without communication.
- However, the merged set of OWNED constraints CANNOT be closed (chain resolution) before making it consistent accros processors. Otherwise, a local owned slave might be constrained by a local ghost slave whose masters are not yet available locally, so the chain resolution would fail (or would be incomplete).
- On the other hand, once the merget set of constraints is made consistent, the chain resolution can be done locally without communication, since every process has a complete local view of the constraint DAG restricted to its local DoFs (owned + ghost + fictitious).

The process has to be "locally merge, then consistent, then locally close".
Thus, we can proceed as follows:

- We now provide a globally consistent `DOF_to_constraint` array, which assigns a constraint source to each DOF (0 for unconstrained). Then `DOF_to_is_slave` is given by `DOF_to_constraint .> 0`. We build the same three global communicators as in Case 1, that is a slave global numbering (sDOF_gids), a free master global numbering (mfdof_gids), and a Dirichlet master global numbering (mddof_gids). Note that these are the final global communicators, shared by all constraint sources, not one per source.
- For each constraint source, we extract the subset of slaves belonging to that source and build the corresponding sub-communicator `csDOF_gids`. This communicator is then provided to the source's callback, which fills up the OWNED constraints for that source.
- We proceed to merging the OWNED constraints from all sources locally, without communication. This may create chains of master-slave relationships across sources. All these masters are local.
- We then perform the same communication step as in Case 1 to make the merged set of constraints consistent across processes.
- At this point, we have a complete, consistent local view of the merged constraint DAG on each process. We can then run `close!` locally on each process to resolve all chains, without any further communication.
- Finally, we reindex to mdof numbering as in Case 1.

## Building-block overview

### Building the global communicators

Currently, we do this in a very similar way to how we build DOF communicators for generic spaces. That is, we use an available `cell_gids` communicator together with a `cell_to_DOFs` table. The key thing here is that the slave, free master, and Dirichlet master split creates  a coloring of the local DOFs (each DOF has only one color), so all three communicators can be built with a single cell-based table communication step.

This step is the one that strictly requires the assumption at the beginning. We cannot account for DOFs that do not belong to the local spaces because they do not belong to the local triangulation.

### Consistent constraints

To obtain consistent constraints accross processes, we need to have a communicator for the slaves (i.e `sDOF_gids`) and the local constraint tables `sDOF_to_DOFs` and `sDOF_to_coeffs` filled with the OWNED constraints.

Additionally, we need some kind of globally consistent key to be able to communicate master DOFs unambiguously across processes. 
To this end, we create a single global numbering of all local DOFs (`DOF_gids`) built as a (local and global) concatenation of the three global communicators (slaves first, then free masters, then Dirichlet masters). Each process can compute this numbering locally without communication.

The procedure is then simple: Map the DOFs to global ids, communicate using `consistent!` to overwrite ghost slave rows with their owners' definitions, then map back to local DOF ids.

After the communication step, we have a consistent set of constraints. However, some ghost constraints may reference master DOFs that do not belong to the local index space. We can these `fictitious` or `remote` DOFs. This means that when mapping back to local DOF ids, we need to keep track and assign new local ids to these fictitious masters, so that they can be referenced in the local constraint tables for chain resolution.

### Reindexing to mDOF numbering

After consistency, we have a complete set of constraints in DOF local numbering. We then need to reindex them to mDOF numbering, which is what the space constructor expects. This is a purely local step, no communication needed. This reindexing also extends the global communicators for free and Dirichlet masters to cover any fictitious masters that were introduced in the consistency step.

This reindexing and extension do not require any communication.

### Local merging of constraints

Merging of constraints from multiple sources is purely local, since we are only merging the OWNED constraints and our assumption guarantees that all masters of owned slaves are local.

The main thing to lookout for is that (of course) all constraints must be defined on the same local index space.
This is also where the assumption is needed: If we allow for remote masters in the owned constraints, the local numbering of these remote masters might be different within each source.

Thus, to remove this assumption we would need to first unify the local index space across sources. This means we would need a global numbering of the original DOFs to be able to identify and match remote masters accross sources.

### Chain resolution (closing) and topological ordering

`close!` can now be run **independently and locally on each process**, with no further communication. The only thing we need is a **global topological ordering** of the DAGs. What I mean by this: 

- Imagine we have a slave `A` that is constraines by two other slaves `B` and `C`, such that `B` and `C` do not depend on each other.
- Because `A` depends on `B` (or `C`), `B` will be resolved BEFORE `A` in ALL processes (since local DAGs are consistent). So this relationship is unambiguous and naturally global.
- However, since `B` and `C` do not depend on each other, their ordering is not fixed by the DAG structure. Then two different processes might resolve `B` before `C` and the other way around, which would lead to different final constraints after closing.

This second case is the one where we need a global tie-breaking rule to ensure that all processes make the same choice. In practice, this is very simple: We just need to provode a globally-consistent key (e.g. global DoF id) to be used within a priority queue to create the topological ordering. This way, all processes will break ties in the same way and end up with the same closed constraints.
