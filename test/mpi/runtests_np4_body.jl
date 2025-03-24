function all_tests(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  t = PArrays.PTimer(ranks,verbose=true)
  PArrays.tic!(t)

  TestApp.GeometryTests.main(distribute,parts)
  PArrays.toc!(t,"Geometry")

  TestApp.CellDataTests.main(distribute,parts)
  PArrays.toc!(t,"CellData")

  TestApp.FESpacesTests.main(distribute,parts)
  PArrays.toc!(t,"FESpaces")

  TestApp.MultiFieldTests.main(distribute,parts)
  PArrays.toc!(t,"MultiField")

  TestApp.PoissonTests.main(distribute,parts)
  PArrays.toc!(t,"Poisson")

  TestApp.PLaplacianTests.main(distribute,parts)
  PArrays.toc!(t,"PLaplacian")

  TestApp.TransientDistributedCellFieldTests.main(distribute,parts)
  PArrays.toc!(t,"TransientDistributedCellField")

  TestApp.TransientMultiFieldDistributedCellFieldTests.main(distribute,parts)
  PArrays.toc!(t,"TransientMultiFieldDistributedCellField")

  TestApp.ZeroMeanFESpacesTests.main(distribute,parts)
  PArrays.toc!(t,"ZeroMeanFESpaces")

  TestApp.PeriodicBCsTests.main(distribute,parts)
  PArrays.toc!(t,"PeriodicBCs")

  TestApp.SurfaceCouplingTests.main(distribute,parts)
  PArrays.toc!(t,"SurfaceCoupling")

  TestApp.HeatEquationTests.main(distribute,parts)
  PArrays.toc!(t,"HeatEquation")

  TestApp.StokesOpenBoundaryTests.main(distribute,parts)
  PArrays.toc!(t,"StokesOpenBoundary")

  if prod(parts) == 4
    TestApp.AdaptivityTests.main(distribute)
    TestApp.AdaptivityCartesianTests.main(distribute)
    TestApp.AdaptivityMultiFieldTests.main(distribute)
    TestApp.AdaptivityUnstructuredTests.main(distribute)
    PArrays.toc!(t,"Adaptivity")
  end

  TestApp.BlockSparseMatrixAssemblersTests.main(distribute,parts)
  PArrays.toc!(t,"BlockSparseMatrixAssemblers")

  if prod(parts) == 4
    TestApp.ConstantFESpacesTests.main(distribute,parts)
    PArrays.toc!(t,"ConstantFESpaces")
  end
  
  if prod(parts) == 4
    TestApp.VisualizationTests.main(distribute,parts)
    PArrays.toc!(t,"Visualization")
  end

  TestApp.AutodiffTests.main(distribute,parts)
  PArrays.toc!(t,"Autodiff")

  display(t)
end
