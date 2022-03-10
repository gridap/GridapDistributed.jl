module TestApp
  include("../../CellDataTests.jl")
  include("../../FESpacesTests.jl")
  include("../../GeometryTests.jl")
  include("../../MultiFieldTests.jl")
  include("../../PLaplacianTests.jl")
  include("../../PoissonTests.jl")
  include("../../PeriodicBCsTests.jl")
  include("../../ODEs/TransientDistributedCellFieldTests.jl")
  include("../../ODEs/TransientMultiFieldDistributedCellFieldTests.jl")
  include("../../ODEs/HeatEquationTests.jl")
  include("../../ODEs/StokesOpenBoundaryTests.jl")
end
