# FESpaces

## Moment-based FE spaces

`raviart_thomas`, `bdm`, `sdiv`, `trimmed_sdiv`, `nedelec`, `scurl`,
`trimmed_scurl`, `rotating_pλ`, `trimmed_pλ` and `CartProd` products of those
are supported.

All of them require the ranks to list the vertices of a shared face in the same
order. That holds for `CartesianDiscreteModel` and for anything built through
`Geometry.restrict`, which inherits the face-vertex orderings from its parent
topology. It is unverified for models whose local topologies are built
independently per rank, such as `GridapP4est` octrees.

```@autodocs
Modules = [GridapDistributed]
Pages   = ["FESpaces.jl","Pullbacks.jl"]
```
