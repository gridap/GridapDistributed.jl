module DistributedCellQuadraturesTests

using Gridap
using Gridap.Geometry: get_cell_id
using GridapDistributed
using Test

subdomains = (2,3)
comm = SequentialCommunicator(subdomains)

domain = (0,1,0,1)
cells = (10,10)
model = CartesianDiscreteModel(comm,subdomains,domain,cells)

trian = Triangulation(model)

degree = 1
quad = CellQuadrature(trian,degree)

integral = integrate( 1 ,trian, quad)

# TODO a more elegant way to filter contributions of ghost cells
sums = ScatteredVector{Float64}(
  comm, integral, model, trian) do part, integral, (model,gids), trian
  i = collect(integral)
  lids = get_cell_id(trian)
  mask = gids.lid_to_owner[lids] .== part
  sum(i[mask])
end

v = sum(gather(sums))
if i_am_master(comm)
  @test v ≈ 1
end

end # module
