///< Include the PyTorch C++ library header.
#include <torch/torch.h>

///< Header guard to prevent multiple inclusions of this file.
#ifndef TRANSFORM_CUH
///< Defines TRANSFORM_CUH to indicate that this header has been included.
#define TRANSFORM_CUH

///< Namespace for transformation functions.
namespace transform_ {
    ///< Calculates the x-component of momentum from transverse momentum (pt) and azimuthal angle (phi).
    torch::Tensor Px(torch::Tensor* pt, torch::Tensor* phi);
    ///< Calculates the y-component of momentum from transverse momentum (pt) and azimuthal angle (phi).
    torch::Tensor Py(torch::Tensor* pt, torch::Tensor* phi);
    ///< Calculates the z-component of momentum from transverse momentum (pt) and pseudorapidity (eta).
    torch::Tensor Pz(torch::Tensor* pt, torch::Tensor* eta);
    ///< Calculates the 3-momentum (Px, Py, Pz) from transverse momentum (pt), pseudorapidity (eta), and azimuthal angle (phi).
    torch::Tensor PxPyPz(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi);
    ///< Calculates the 4-momentum (Px, Py, Pz, E) from transverse momentum (pt), pseudorapidity (eta), azimuthal angle (phi), and energy.
    torch::Tensor PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* energy);
    ///< Calculates the 3-momentum (Px, Py, Pz) from a 4-momentum tensor (pmu).
    torch::Tensor PxPyPz(torch::Tensor* pmu);
    ///< Calculates the 4-momentum (Px, Py, Pz, E) from a 4-momentum tensor (pmu) - likely extracts or ensures all four components.
    torch::Tensor PxPyPzE(torch::Tensor* pmu);

    ///< Calculates transverse momentum (pt), pseudorapidity (eta), and azimuthal angle (phi) from a Cartesian 4-momentum tensor (pmc).
    torch::Tensor PtEtaPhi(torch::Tensor* pmc);
    ///< Calculates transverse momentum (pt), pseudorapidity (eta), and azimuthal angle (phi) from Cartesian momentum components (px, py, pz).
    torch::Tensor PtEtaPhi(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

    ///< Calculates transverse momentum (pt), pseudorapidity (eta), azimuthal angle (phi), and energy (E) from a Cartesian 4-momentum tensor (pmc).
    torch::Tensor PtEtaPhiE(torch::Tensor* pmc);
    ///< Calculates transverse momentum (pt), pseudorapidity (eta), azimuthal angle (phi), and energy (E) from Cartesian momentum components (px, py, pz) and energy (e).
    torch::Tensor PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

    ///< Calculates transverse momentum (Pt) from Cartesian momentum components (px, py).
    torch::Tensor Pt(torch::Tensor* px, torch::Tensor* py);
    ///< Calculates azimuthal angle (Phi) from Cartesian momentum components (px, py).
    torch::Tensor Phi(torch::Tensor* px, torch::Tensor* py);

    ///< Calculates pseudorapidity (Eta) from a Cartesian 4-momentum tensor (pmc).
    torch::Tensor Eta(torch::Tensor* pmc); 
    ///< Calculates pseudorapidity (Eta) from Cartesian momentum components (px, py, pz).
    torch::Tensor Eta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

///< End of the transform_ namespace.
}

///< End of the header guard.
#endif
