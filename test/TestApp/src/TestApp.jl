module TestApp
  include("../../CellDataTests.jl")
  include("../../FESpacesTests.jl")
  include("../../GeometryTests.jl")
  include("../../MultiFieldTests.jl")
  include("../../PLaplacianTests.jl")
  include("../../PoissonTests.jl")
  include("../../PeriodicBCsTests.jl")
  include("../../GridapODEs/TransientDistributedCellFieldTests.jl")
  include("../../GridapODEs/TransientMultiFieldDistributedCellFieldTests.jl")
  include("../../GridapODEs/HeatEquationTests.jl")
  include("../../GridapODEs/StokesOpenBoundaryTests.jl")
end
