module DistributedCellQuadraturesTests

using Gridap
using GridapDistributed
using Test

subdomains = (2,3)
comm = SequentialCommunicator(subdomains)

domain = (0,1,0,1)
cells = (10,10)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

trian = Triangulation(model)

trian = remove_ghost_cells(trian,model)
#trian = include_ghost_cells(trian)

degree = 1
quad = CellQuadrature(trian,degree)

integral = integrate( 1 ,trian, quad)

# TODO better API?
sums = ScatteredVector(integral) do part, integral
  sum(integral)
end

v = sum(gather(sums))
if i_am_master(comm)
  @test v ≈ 1
end

u(x) = x[2]
integral = integrate( u ,trian, quad)

# TODO better API?
sums = ScatteredVector(integral) do part, integral
  sum(integral)
end

v = sum(gather(sums))
if i_am_master(comm)
  @test v ≈ 0.5
end

end # module
