function all_tests(distribute, parts)
  TESTCASE = get(ENV, "TESTCASE", "all")

  ranks = distribute(LinearIndices((prod(parts),)))

  t = PArrays.PTimer(ranks, verbose=true)
  PArrays.tic!(t)

  if TESTCASE ∈ ("all", "mpi-geometry")
    GeometryTests.main(distribute, parts)
    PArrays.toc!(t, "Geometry")

    CellDataTests.main(distribute, parts)
    PArrays.toc!(t, "CellData")
  end

  if TESTCASE ∈ ("all", "mpi-fespaces")
    FESpacesTests.main(distribute, parts)
    PArrays.toc!(t, "FESpaces")

    MultiFieldTests.main(distribute, parts)
    PArrays.toc!(t, "MultiField")

    ZeroMeanFESpacesTests.main(distribute, parts)
    PArrays.toc!(t, "ZeroMeanFESpaces")

    PeriodicBCsTests.main(distribute, parts)
    PArrays.toc!(t, "PeriodicBCs")

    if prod(parts) == 4
      ConstantFESpacesTests.main(distribute, parts)
      PArrays.toc!(t, "ConstantFESpaces")
    end
  end

  if TESTCASE ∈ ("all", "mpi-physics")
    PoissonTests.main(distribute, parts)
    PArrays.toc!(t, "Poisson")

    PLaplacianTests.main(distribute, parts)
    PArrays.toc!(t, "PLaplacian")

    SurfaceCouplingTests.main(distribute, parts)
    PArrays.toc!(t, "SurfaceCoupling")

    StokesOpenBoundaryTests.main(distribute, parts)
    PArrays.toc!(t, "StokesOpenBoundary")

    StokesHdivDGTests.main(distribute, parts)
    PArrays.toc!(t, "StokesHdivDG")
  end

  if TESTCASE ∈ ("all", "mpi-transient")
    TransientDistributedCellFieldTests.main(distribute, parts)
    PArrays.toc!(t, "TransientDistributedCellField")

    TransientMultiFieldDistributedCellFieldTests.main(distribute, parts)
    PArrays.toc!(t, "TransientMultiFieldDistributedCellField")

    HeatEquationTests.main(distribute, parts)
    PArrays.toc!(t, "HeatEquation")
  end

  if TESTCASE ∈ ("all", "mpi-adaptivity")
    if prod(parts) == 4
      AdaptivityTests.main(distribute)
      AdaptivityCartesianTests.main(distribute)
      AdaptivityMultiFieldTests.main(distribute)
      AdaptivityUnstructuredTests.main(distribute)
      PolytopalCoarseningTests.main(distribute, parts)
      PArrays.toc!(t, "Adaptivity")
    end
  end

  if TESTCASE ∈ ("all", "mpi-misc")
    BlockSparseMatrixAssemblersTests.main(distribute, parts)
    BlockPartitionedArraysTests.main(distribute, parts)
    PArrays.toc!(t, "BlockSparseMatrixAssemblers")

    if prod(parts) == 4
      VisualizationTests.main(distribute, parts)
      PArrays.toc!(t, "Visualization")
    end

    AutodiffTests.main(distribute, parts)
    PArrays.toc!(t, "Autodiff")

    if prod(parts) == 4
      MacroDiscreteModelsTests.main(distribute, parts)
      PArrays.toc!(t, "MacroDiscreteModels")
    end
  end

  isempty(t.timings) || display(t)
end
