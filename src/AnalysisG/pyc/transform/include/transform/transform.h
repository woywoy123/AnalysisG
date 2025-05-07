/**
 * @file transform.h
 * @brief Provides transformation functions for momentum calculations in C++.
 */

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <torch/torch.h> ///< Includes PyTorch C++ headers for tensor operations.

/**
 * @brief Namespace for transformation functions.
 */
namespace transform_ {

/**
 * @brief Computes the x-component of momentum (Px).
 *
 * @param pt Input tensor of transverse momentum values.
 * @param phi Input tensor of azimuthal angle values.
 * @return A tensor containing the x-component of momentum.
 */
torch::Tensor Px(torch::Tensor* pt, torch::Tensor* phi);

/**
 * @brief Computes the y-component of momentum (Py).
 *
 * @param pt Input tensor of transverse momentum values.
 * @param phi Input tensor of azimuthal angle values.
 * @return A tensor containing the y-component of momentum.
 */
torch::Tensor Py(torch::Tensor* pt, torch::Tensor* phi);

/**
 * @brief Computes the z-component of momentum (Pz).
 *
 * @param pt Input tensor of transverse momentum values.
 * @param eta Input tensor of pseudorapidity values.
 * @return A tensor containing the z-component of momentum.
 */
torch::Tensor Pz(torch::Tensor* pt, torch::Tensor* eta);

/**
 * @brief Computes the 3-momentum (Px, Py, Pz).
 *
 * @param pt Input tensor of transverse momentum values.
 * @param eta Input tensor of pseudorapidity values.
 * @param phi Input tensor of azimuthal angle values.
 * @return A tensor containing the 3-momentum.
 */
torch::Tensor PxPyPz(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi);

/**
 * @brief Computes the 4-momentum (Px, Py, Pz, E).
 *
 * @param pt Input tensor of transverse momentum values.
 * @param eta Input tensor of pseudorapidity values.
 * @param phi Input tensor of azimuthal angle values.
 * @param energy Input tensor of energy values.
 * @return A tensor containing the 4-momentum.
 */
torch::Tensor PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* energy);

/**
 * @brief Computes the 3-momentum (Px, Py, Pz) from a 4-momentum tensor.
 *
 * @param pmu Input tensor of 4-momentum values.
 * @return A tensor containing the 3-momentum.
 */
torch::Tensor PxPyPz(torch::Tensor* pmu);

/**
 * @brief Computes the 4-momentum (Px, Py, Pz, E) from a 4-momentum tensor.
 *
 * @param pmu Input tensor of 4-momentum values.
 * @return A tensor containing the 4-momentum.
 */
torch::Tensor PxPyPzE(torch::Tensor* pmu);

/**
 * @brief Computes the transverse momentum (Pt).
 *
 * @param px Input tensor of x-component of momentum.
 * @param py Input tensor of y-component of momentum.
 * @return A tensor containing the transverse momentum.
 */
torch::Tensor Pt(torch::Tensor* px, torch::Tensor* py);

/**
 * @brief Computes the transverse momentum (Pt) and pseudorapidity (Eta).
 *
 * @param pt Input tensor of transverse momentum values.
 * @param pz Input tensor of z-component of momentum.
 * @return A tensor containing the transverse momentum and pseudorapidity.
 */
torch::Tensor PtEta(torch::Tensor* pt, torch::Tensor* pz);

/**
 * @brief Computes the azimuthal angle (Phi) from a momentum tensor.
 *
 * @param pmc Input tensor of momentum values.
 * @return A tensor containing the azimuthal angle.
 */
torch::Tensor Phi(torch::Tensor* pmc);

/**
 * @brief Computes the azimuthal angle (Phi) from x and y components of momentum.
 *
 * @param px Input tensor of x-component of momentum.
 * @param py Input tensor of y-component of momentum.
 * @return A tensor containing the azimuthal angle.
 */
torch::Tensor Phi(torch::Tensor* px, torch::Tensor* py);

/**
 * @brief Computes the pseudorapidity (Eta) from a momentum tensor.
 *
 * @param pmc Input tensor of momentum values.
 * @return A tensor containing the pseudorapidity.
 */
torch::Tensor Eta(torch::Tensor* pmc);

/**
 * @brief Computes the pseudorapidity (Eta) from x, y, and z components of momentum.
 *
 * @param px Input tensor of x-component of momentum.
 * @param py Input tensor of y-component of momentum.
 * @param pz Input tensor of z-component of momentum.
 * @return A tensor containing the pseudorapidity.
 */
torch::Tensor Eta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes the transverse momentum (Pt), pseudorapidity (Eta), and azimuthal angle (Phi) from a momentum tensor.
 *
 * @param pmc Input tensor of momentum values.
 * @return A tensor containing Pt, Eta, and Phi.
 */
torch::Tensor PtEtaPhi(torch::Tensor* pmc);

/**
 * @brief Computes the transverse momentum (Pt), pseudorapidity (Eta), azimuthal angle (Phi), and energy (E) from a momentum tensor.
 *
 * @param pmc Input tensor of momentum values.
 * @return A tensor containing Pt, Eta, Phi, and E.
 */
torch::Tensor PtEtaPhiE(torch::Tensor* pmc);

/**
 * @brief Computes the transverse momentum (Pt), pseudorapidity (Eta), and azimuthal angle (Phi) from x, y, and z components of momentum.
 *
 * @param px Input tensor of x-component of momentum.
 * @param py Input tensor of y-component of momentum.
 * @param pz Input tensor of z-component of momentum.
 * @return A tensor containing Pt, Eta, and Phi.
 */
torch::Tensor PtEtaPhi(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes the transverse momentum (Pt), pseudorapidity (Eta), azimuthal angle (Phi), and energy (E) from x, y, z components of momentum and energy.
 *
 * @param px Input tensor of x-component of momentum.
 * @param py Input tensor of y-component of momentum.
 * @param pz Input tensor of z-component of momentum.
 * @param e Input tensor of energy values.
 * @return A tensor containing Pt, Eta, Phi, and E.
 */
torch::Tensor PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

} // namespace transform_

#endif // TRANSFORM_H
