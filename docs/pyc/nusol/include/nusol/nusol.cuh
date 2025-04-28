/**
 * @file nusol.cuh
 * @brief Defines the interface for the `nusol_` namespace, providing a suite of CUDA-accelerated functions for neutrino momentum reconstruction in high-energy physics analyses, particularly focused on decays involving W bosons (e.g., top quark decays). These functions leverage PyTorch tensors for efficient data handling and GPU computation.
 */

#ifndef CUNUSOL_CUDA_H
#define CUNUSOL_CUDA_H

#include <map>          // Required for std::map, used as the return type for results.
#include <string>       // Required for std::string, used as keys in the result map.
#include <torch/torch.h> // Required for torch::Tensor, the primary data structure for inputs and outputs.

/**
 * @namespace nusol_
 * @brief Encapsulates CUDA-accelerated algorithms for reconstructing neutrino kinematics in particle physics events.
 * @details This namespace provides functions primarily designed for reconstructing the four-momentum of neutrinos produced in W boson decays, a common task in analyses involving top quarks (e.g., ttbar events). The algorithms range from analytical solutions for single W decays to numerical methods for dileptonic systems and combinatorial approaches for complex event topologies. All functions operate on PyTorch tensors residing on the GPU, enabling high-performance computation. The results are consistently returned as `std::map<std::string, torch::Tensor>`, allowing flexible access to various calculated quantities.
 */
namespace nusol_ {
        /**
         * @brief Computes essential kinematic coefficients (the "base matrix") for the analytical neutrino reconstruction algorithm.
         * @details This function calculates intermediate values derived from the kinematics of the b-quark and the charged lepton originating from a W boson decay, along with the assumed masses of the top quark, W boson, and neutrino. These values often represent coefficients of quadratic equations or parameters of conic sections used to constrain the neutrino's longitudinal momentum (pz). The calculation is performed element-wise for a batch of events.
         *
         * @param pmc_b A pointer to a GPU-resident `torch::Tensor` holding the four-momenta (Px, Py, Pz, E) of the b-quarks.
         *              - Shape: `[N, 4]`, where N is the number of events.
         *              - Data Type: Typically `torch::kFloat64` or `torch::kFloat32`.
         *              - Device: Must be CUDA.
         * @param pmc_mu A pointer to a GPU-resident `torch::Tensor` holding the four-momenta (Px, Py, Pz, E) of the charged leptons (e.g., muons or electrons).
         *               - Shape: `[N, 4]`.
         *               - Data Type: Typically `torch::kFloat64` or `torch::kFloat32`.
         *               - Device: Must be CUDA.
         * @param masses A pointer to a GPU-resident `torch::Tensor` containing the masses for the calculation: [mass_top, mass_W, mass_neutrino].
         *               - Shape: `[N, 3]` (per-event masses) or `[3]` (broadcasted across events).
         *               - Data Type: Typically `torch::kFloat64` or `torch::kFloat32`.
         *               - Device: Must be CUDA.
         * @return A `std::map<std::string, torch::Tensor>` where keys identify specific calculated coefficients (e.g., "a", "b", "c", "Sx", "Sy", "Sz", "Ex", "Ey", "Ez", "A", "B", "C", "D", "F", "G", "H", "I", "J", "K", "L",
