# Assembly strategies

GridapDistributed offers several assembly strategies for distributed linear systems. These strategies modify the ghost layout for the rows and columns of the assembled matric and vector. Depending on your usecase, one strategy may be more convenient than the others.

## SubAssembledRows

!!! info
    - **Main idea:** Both columns and rows are ghosted, whith (potentially) different ghost layouts. Assembly is costly but matrix-vector products are cheap.
    - **Pros:** Matrix-vector product fills both owned and ghost rows of the output vector. Communication is therefore not required to make the output vector consistent.
    - **Cons:** Communication is required to assemble the matrix and vector.
    - **Use cases:** Default assembly strategy.

- Each processor integrates over the **owned cells**, i.e there are no duplicated cell contributions. However, processors do not hold all the contributions they need to assemble their matrix and vector.
- 

## FullyAssembledRows

!!! info
    - **Main idea:** Columns are ghosted, but rows ownly contain owned indices. Assembly is cheap but matrix-vector products are costly.
    - **Pros:** Assembly is local, i.e no communication is required. Column vectors can also be used as row vectors.
    - **Cons:** Matrix-vector product only fills the owned rows of the output vector. Communication is therefore required to make the output vector consistent.
    - **Use cases:** This is the strategy used by PETSc. You should also use this strategy if you plan to feed back output row vectors as input column vectors during successive matrix-vector products.

- Each processor integrates over **all it's local (owned + ghost) cells**, i.e contributions for interface cells are duplicated. This implies that each processor has access to **all** the contributions for its **owned dofs** without need for any communication.
- Contributions whose row index is not owned by the processor are discarded, while owned rows can be fully assembled without any communication.

## FEConsistentAssembly

!!! info
    - **Main idea:** Same as `FullyAssembledRows` but the ghost layout for the columns is the same as the original `FESpace` ghost layout.
    - **Pros:** Assembly is local, i.e no communication is required. DoF `PVector`s from the `FESpace` can be used as column and row vectors for the matrix (like in serial).
    - **Cons:** Matrix-vector product only fills the owned rows of the output vector. Communication is therefore required to make the output vector consistent.
    - **Use cases:** You should use this strategy if you are constantly creating `FEFunction`s with vectors coming from the linear system (and viceversa). This is quite typical for geometric solvers.
