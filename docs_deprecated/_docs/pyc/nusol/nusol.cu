/**
 * @file nusol.cu
 * @brief This file contains the CUDA implementation for reconstructing neutrino four-momenta in particle physics events, specifically focusing on combinatorial methods.
 *
 * The core functionality lies within the `nusol_::combinatorial` function, which explores various combinations
 * of leptons and b-quarks within an event to find the most likely neutrino solutions consistent with kinematic constraints
 * (like W and top quark masses) and the measured missing transverse energy (MET).
 */

/**
 * @namespace nusol_
 * @brief Provides functions and utilities for neutrino solution reconstruction using CUDA.
 *
 * This namespace encapsulates the logic for determining the four-momenta of neutrinos, which are particles
 * that typically escape detection in particle colliders. The reconstruction relies on kinematic principles
 * and the properties of particle decays (e.g., top quark decaying to W boson and b-quark, W boson decaying to lepton and neutrino).
 */
namespace nusol_ {

/**
 * @brief Reconstructs neutrino four-momenta using a combinatorial approach on the GPU.
 *
 * This function implements a combinatorial algorithm to solve for the four-momenta of two neutrinos
 * produced in events, typically assumed to originate from the decay of two W bosons (e.g., in ttbar events).
 * It iterates through all possible pairings of identified leptons (electrons or muons) and b-quarks within each event
 * in a batch. For each combination, it attempts to solve the kinematic equations constrained by the known W boson mass (mW)
 * and top quark mass (mT), considering the measured missing transverse energy (MET).
 *
 * The function employs a numerical minimization technique (potentially iterative) to find the neutrino momenta
 * that best satisfy the constraints and minimize a distance metric, which quantifies the quality of the solution.
 * The process is parallelized across events and combinations using CUDA.
 *
 * @param edge_index A pointer to a `torch::Tensor` representing the graph structure of the events, if applicable.
 *                   This might define relationships between particles. Shape: `(num_edges, 2)`. Expected dtype: `torch::kInt64`.
 *                   *Note: The exact usage depends on the downstream implementation details.*
 * @param batch A pointer to a `torch::Tensor` mapping each particle to its corresponding event index within the batch.
 *              This is crucial for processing multiple events simultaneously. Shape: `(num_particles,)`. Expected dtype: `torch::kInt64`.
 * @param pmc A pointer to a `torch::Tensor` containing the measured four-momenta (Px, Py, Pz, E) of the relevant particles
 *            (leptons and b-quarks) in the batch. Shape: `(num_particles, 4)`. Expected dtype: `torch::kFloat64` or `torch::kFloat32`.
 * @param pid A pointer to a `torch::Tensor` containing particle identifiers. This is used to distinguish between leptons and b-quarks,
 *            and potentially between different types of leptons (electrons/muons) or charges.
 *            Shape: `(num_particles, N_pid_features)`. The exact structure depends on the encoding scheme. Expected dtype: `torch::kInt64` or similar.
 * @param met_xy A pointer to a `torch::Tensor` containing the measured missing transverse energy components (METx, METy) for each event.
 *               MET represents the momentum imbalance in the plane transverse to the beam direction, attributed primarily to undetected neutrinos.
 *               Shape: `(num_events, 2)`. Expected dtype: `torch::kFloat64` or `torch::kFloat32`.
 * @param mT The assumed mass of the top quark in GeV. This is used as a constraint in the kinematic fit, assuming the leptons and b-quarks originate from top decays.
 * @param mW The assumed mass of the W boson in GeV. This is used as a constraint, assuming the lepton-neutrino pairs originate from W decays.
 * @param null A parameter likely related to handling ambiguities or numerical stability in the underlying solver equations. It might represent a tolerance, a regularization term, or a specific value used in the quadratic equation solutions for neutrino pz.
 * @param perturb A small value used for numerical perturbation, potentially during the minimization process or to handle degenerate cases in the solver.
 * @param steps The maximum number of iterations allowed for the numerical minimization procedure used to refine the neutrino solutions for each combination.
 * @param gev A boolean flag indicating the units of the input masses (`mT`, `mW`) and potentially the particle momenta (`pmc`).
 *            If `true`, units are assumed to be Giga-electron Volts (GeV). If `false`, Mega-electron Volts (MeV) might be assumed (though GeV is standard in this context).
 *
 * @return A `std::map<std::string, torch::Tensor>` containing the results of the combinatorial reconstruction. The map holds tensors associated with the *best* solution found for each event after evaluating all valid combinations.
 *         - `"distances"`: (`torch::Tensor`, shape: `(num_events,)`, dtype: float) - A metric indicating the quality of the best solution found for each event. Lower values typically signify a better fit to the kinematic constraints.
 *         - `"l1"`: (`torch::Tensor`, shape: `(num_events,)`, dtype: int64) - The index (in the original `pmc`/`pid` tensors) of the first lepton used in the best solution for each event.
 *         - `"l2"`: (`torch::Tensor`, shape: `(num_events,)`, dtype: int64) - The index of the second lepton used in the best solution for each event.
 *         - `"b1"`: (`torch::Tensor`, shape: `(num_events,)`, dtype: int64) - The index of the first b-quark used in the best solution for each event.
 *         - `"b2"`: (`torch::Tensor`, shape: `(num_events,)`, dtype: int64) - The index of the second b-quark used in the best solution for each event.
 *         - `"nu1"`: (`torch::Tensor`, shape: `(num_events, 4)`, dtype: float) - The reconstructed four-momentum (Px, Py, Pz, E) of the first neutrino corresponding to the best solution for each event.
 *         - `"nu2"`: (`torch::Tensor`, shape: `(num_events, 4)`, dtype: float) - The reconstructed four-momentum (Px, Py, Pz, E) of the second neutrino corresponding to the best solution for each event.
 *         - `"msk"`: (`torch::Tensor`, shape: `(num_combinations,)`, dtype: bool or int) - A mask tensor, potentially indicating which combinations across all events were considered valid or yielded a successful solution before the final selection. The exact size `num_combinations` depends on the total number of valid lepton/b-quark pairings across the batch.
 *
 * @note This function is expected to be called from a C++ environment that interfaces with PyTorch (using libtorch) and CUDA.
 * @warning The input tensors (`edge_index`, `batch`, `pmc`, `pid`, `met_xy`) must reside on the CUDA device where the computation is intended to run. The function likely performs internal checks or assumes this precondition.
 */
std::map<std::string, torch::Tensor> combinatorial(
    torch::Tensor* edge_index, torch::Tensor* batch, torch::Tensor* pmc,
    torch::Tensor* pid, torch::Tensor* met_xy,
    double  mT, double mW, double null, double perturb, long steps, bool gev
);

} // namespace nusol_
