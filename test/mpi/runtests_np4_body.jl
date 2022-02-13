function all_tests(parts)
   display(parts)

   t = PArrays.PTimer(parts,verbose=true)
   PArrays.tic!(t)

   TestApp.GeometryTests.main(parts)
   PArrays.toc!(t,"Geometry")

   TestApp.CellDataTests.main(parts)
   PArrays.toc!(t,"CellData")

   TestApp.FESpacesTests.main(parts)
   PArrays.toc!(t,"FESpaces")

   TestApp.MultiFieldTests.main(parts)
   PArrays.toc!(t,"MultiField")

   TestApp.PoissonTests.main(parts)
   PArrays.toc!(t,"Poisson")

   TestApp.PLaplacianTests.main(parts)
   PArrays.toc!(t,"PLaplacian")

   TestApp.TransientDistributedCellFieldTests.main(parts)
   PArrays.toc!(t,"TransientDistributedCellField")

   TestApp.TransientMultiFieldDistributedCellFieldTests.main(parts)
   PArrays.toc!(t,"TransientMultiFieldDistributedCellField")

  display(t)
end
