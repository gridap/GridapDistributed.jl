# Functions for transient FE spaces

Fields.evaluate(U::DistributedSingleFieldFESpace,t::Nothing) = U

(U::DistributedSingleFieldFESpace)(t) = U

ODEs.∂t(U::DistributedSingleFieldFESpace) = HomogeneousTrialFESpace(U)
ODEs.∂t(U::DistributedMultiFieldFESpace) = MultiFieldFESpace(∂t.(U.field_fe_space))
ODEs.∂tt(U::DistributedSingleFieldFESpace) = HomogeneousTrialFESpace(U)
ODEs.∂tt(U::DistributedMultiFieldFESpace) = MultiFieldFESpace(∂tt.(U.field_fe_spaces))

function TransientMultiFieldFESpace(spaces::Vector{<:DistributedSingleFieldFESpace})
  MultiFieldFESpace(spaces)
end
