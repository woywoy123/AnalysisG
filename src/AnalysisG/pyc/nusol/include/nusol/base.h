/**
 * @file base.h
 * @brief Provides base definitions for the nusol module in C++.
 */

#ifndef NUSOL_BASE_H
#define NUSOL_BASE_H

#include <torch/torch.h> ///< Includes PyTorch C++ headers for tensor operations.

/**
 * @brief Namespace for neutrino solution (nusol) base functions.
 */
namespace nusol {

/**
 * @brief Computes the base matrix for neutrino momentum solutions.
 *
 * @param pmc_b Input tensor for b-quark momenta.
 * @param pmc_mu Input tensor for muon momenta.
 * @param masses Input tensor for mass values.
 * @return A tensor containing the base matrix.
 */
torch::Tensor compute_base_matrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses);

/**
 * @brief Solves for neutrino momentum using the base matrix.
 *
 * @param base_matrix Input tensor containing the base matrix.
 * @param met Input tensor for missing transverse energy.
 * @return A tensor containing the neutrino momentum solutions.
 */
torch::Tensor solve_neutrino_momentum(torch::Tensor base_matrix, torch::Tensor met);

} // namespace nusol

#endif // NUSOL_BASE_H
