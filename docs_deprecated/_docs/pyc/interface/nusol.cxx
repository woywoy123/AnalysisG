/**
 * @brief Calculates the base matrix used in neutrino reconstruction algorithms.
 *
 * This function computes a base matrix derived from the four-momenta of the b-quark and the muon,
 * along with their associated masses. This matrix is often a preliminary step in solving
 * for the neutrino momentum in top quark decays (t -> Wb -> lvb).
 *
 * @param pmc_b A torch::Tensor representing the four-momenta (Px, Py, Pz, E) of the b-quarks.
 *              Expected shape: [N, 4], where N is the number of events.
 * @param pmc_mu A torch::Tensor representing the four-momenta (Px, Py, Pz, E) of the muons.
 *               Expected shape: [N, 4], where N is the number of events.
 * @param masses A torch::Tensor containing the relevant particle masses, typically
 *               [mass_top, mass_W, mass_b, mass_muon, mass_neutrino].
 *               Expected shape: [N, 5] or broadcastable to it.
 * @return A torch::Dict<std::string, torch::Tensor> containing the calculated base matrix components.
 *         The specific keys and tensor structures within the dictionary depend on the underlying
 *         implementation of the neutrino reconstruction algorithm.
 */
torch::Dict<std::string, torch::Tensor> BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses);

/**
 * @brief Reconstructs the four-momentum of a single neutrino in an event.
 *
 * This function implements a neutrino reconstruction algorithm (likely based on kinematic constraints
 * from W boson decay) to determine the neutrino's four-momentum (Px, Py, Pz, E). It utilizes the
 * four-momenta of the associated b-quark and lepton (muon in this case), the measured missing
 * transverse energy (MET), particle masses, and potentially detector resolution information (sigma).
 *
 * @param pmc_b A torch::Tensor representing the four-momenta (Px, Py, Pz, E) of the b-quarks.
 *              Expected shape: [N, 4].
 * @param pmc_mu A torch::Tensor representing the four-momenta (Px, Py, Pz, E) of the muons.
 *               Expected shape: [N, 4].
 * @param met_xy A torch::Tensor representing the missing transverse energy components (METx, METy).
 *               Expected shape: [N, 2].
 * @param masses A torch::Tensor containing the relevant particle masses (e.g., W boson mass, top quark mass).
 *               Expected shape: [N, k] or broadcastable, where k depends on the algorithm.
 * @param sigma A torch::Tensor representing uncertainties or resolution parameters used in the reconstruction,
 *              potentially related to MET or jet/lepton measurements. The exact meaning depends on the algorithm.
 *              Expected shape: [N, m] or broadcastable.
 * @param null A double value used to represent cases where the reconstruction fails or yields non-physical results.
 * @return A torch::Dict<std::string, torch::Tensor> containing the reconstructed neutrino four-momentum components
 *         (e.g., "Nu_Px", "Nu_Py", "Nu_Pz", "Nu_E") and potentially other information like solution quality flags.
 *         Failed reconstructions might be indicated by the `null` value.
 */
torch::Dict<std::string, torch::Tensor> Nu(
    torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor met_xy,
    torch::Tensor masses, torch::Tensor sigma, double null
);

/**
 * @brief Reconstructs the four-momenta of two neutrinos simultaneously, typically in dileptonic ttbar events.
 *
 * This function tackles the more complex scenario of reconstructing two neutrinos, often encountered
 * in events like ttbar -> (W+ b) (W- bbar) -> (l+ nu b) (l- nubar bbar). It uses the four-momenta
 * of two b-quarks and two leptons, the total MET, and assumes a common set of mass constraints
 * (e.g., both top quarks and W bosons have the same mass). The reconstruction often involves
 * numerical methods or iterative algorithms.
 *
 * @param pmc_b1 A torch::Tensor for the four-momenta of the first b-quark. Expected shape: [N, 4].
 * @param pmc_b2 A torch::Tensor for the four-momenta of the second b-quark. Expected shape: [N, 4].
 * @param pmc_l1 A torch::Tensor for the four-momenta of the first lepton. Expected shape: [N, 4].
 * @param pmc_l2 A torch::Tensor for the four-momenta of the second lepton. Expected shape: [N, 4].
 * @param met_xy A torch::Tensor for the total missing transverse energy (METx, METy). Expected shape: [N, 2].
 * @param masses A torch::Tensor containing the relevant particle masses (e.g., top, W). Expected shape: [N, k] or broadcastable.
 * @param null A double value used to represent failed reconstruction attempts.
 * @param step A double representing the step size parameter for the numerical solver/optimizer used.
 * @param tolerance A double representing the convergence tolerance for the numerical solver/optimizer.
 * @param timeout An unsigned integer specifying the maximum number of iterations or time limit for the solver.
 * @return A torch::Dict<std::string, torch::Tensor> containing the reconstructed four-momenta for both neutrinos
 *         (e.g., "Nu1_Px", "Nu1_Py", ..., "Nu2_Pz", "Nu2_E") and potentially solution status flags.
 */
torch::Dict<std::string, torch::Tensor> NuNu(
    torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2,
    torch::Tensor met_xy, torch::Tensor masses, double null,
    const double step, const double tolerance, const unsigned int timeout
);

/**
 * @brief Reconstructs the four-momenta of two neutrinos simultaneously with potentially different mass hypotheses for the two decay chains.
 *
 * This is an overload of the NuNu function, allowing for separate mass inputs for the two
 * top/W decay chains. This can be useful for exploring systematic uncertainties or alternative physics models
 * where the masses might differ.
 *
 * @param pmc_b1 A torch::Tensor for the four-momenta of the first b-quark. Expected shape: [N, 4].
 * @param pmc_b2 A torch::Tensor for the four-momenta of the second b-quark. Expected shape: [N, 4].
 * @param pmc_l1 A torch::Tensor for the four-momenta of the first lepton. Expected shape: [N, 4].
 * @param pmc_l2 A torch::Tensor for the four-momenta of the second lepton. Expected shape: [N, 4].
 * @param met_xy A torch::Tensor for the total missing transverse energy (METx, METy). Expected shape: [N, 2].
 * @param null A double value used to represent failed reconstruction attempts.
 * @param mass1 A torch::Tensor containing the masses for the first decay chain (e.g., top1, W1). Expected shape: [N, k] or broadcastable.
 * @param mass2 A torch::Tensor containing the masses for the second decay chain (e.g., top2, W2). Expected shape: [N, k] or broadcastable.
 * @param step A double representing the step size parameter for the numerical solver/optimizer used.
 * @param tolerance A double representing the convergence tolerance for the numerical solver/optimizer.
 * @param timeout An unsigned integer specifying the maximum number of iterations or time limit for the solver.
 * @return A torch::Dict<std::string, torch::Tensor> containing the reconstructed four-momenta for both neutrinos
 *         and potentially solution status flags.
 */
torch::Dict<std::string, torch::Tensor> NuNu(
    torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2,
    torch::Tensor met_xy, double null, torch::Tensor mass1, torch::Tensor mass2,
    const double step, const double tolerance, const unsigned int timeout
);

/**
 * @brief Performs a combinatorial neutrino reconstruction analysis, likely for ttbar events, using tensor inputs.
 *
 * This function implements a combinatorial approach where different assignments of jets and leptons
 * to the decay products of top quarks are tested. It attempts to reconstruct the neutrinos for each
 * combination and potentially selects the best combination based on kinematic consistency or likelihood.
 * This version operates directly on torch tensors, suitable for batch processing or integration with ML frameworks.
 *
 * @param edge_index A torch::Tensor likely representing the graph structure or pairings between particles in events.
 *                   The exact format depends on the graph representation used (e.g., adjacency list for pairings).
 * @param batch A torch::Tensor indicating which event each particle belongs to, used for batch processing. Shape: [num_particles].
 * @param pmc A torch::Tensor containing the four-momenta of all relevant particles (leptons, jets) in the batch.
 *            Shape: [num_particles, 4].
 * @param pid A torch::Tensor containing the particle IDs (PDG ID) for the particles in `pmc`. Shape: [num_particles].
 * @param met_xy A torch::Tensor containing the MET components (METx, METy) for each event. Shape: [num_events, 2].
 * @param mT A double representing the transverse mass constraint, possibly used as a cut or in a likelihood calculation.
 *           Typically the top quark mass.
 * @param mW A double representing the W boson mass constraint, used in the kinematic reconstruction.
 * @param null A double value used to represent invalid or failed combinations/reconstructions.
 * @param perturb A double factor used potentially to perturb input kinematics (e.g., MET, jet energies)
 *                to estimate uncertainties or improve solver stability.
 * @param steps A long integer specifying the number of iterations or steps, possibly related to perturbation or optimization.
 * @param gev A boolean flag indicating whether the input momenta/masses are in GeV (true) or MeV (false).
 * @return A torch::Dict<std::string, torch::Tensor> containing the results of the combinatorial analysis. This could include
 *         the reconstructed neutrino momenta for the best combination, likelihood scores, indices of the selected particles, etc.
 */
torch::Dict<std::string, torch::Tensor> combinatorial(
    torch::Tensor edge_index, torch::Tensor batch, torch::Tensor pmc, torch::Tensor pid, torch::Tensor met_xy,
    double mT, double mW, double null, double perturb, long steps, bool gev
);

/**
 * @brief Performs a combinatorial analysis to pair neutrinos with quarks and leptons using standard C++ containers.
 *
 * This function provides an interface for combinatorial neutrino reconstruction using standard C++ vectors
 * and custom particle objects (`particle_template`). It takes MET information and lists of particles per event,
 * explores possible assignments (e.g., which jet is b1, which lepton is l1), performs neutrino reconstruction
 * for valid combinations, and returns the reconstructed neutrino pairs.
 *
 * @param met_ A std::vector<double> containing the magnitude of the missing transverse energy (MET) for each event.
 * @param phi_ A std::vector<double> containing the azimuthal angle (phi) of the MET vector for each event.
 * @param particles A std::vector<std::vector<particle_template*>> where each inner vector contains pointers
 *                  to the particle objects (jets, leptons) for a single event.
 * @param dev A std::string potentially specifying the compute device ("cpu", "cuda") or method, although its usage
 *            is not fully clear from the signature alone in this C++ context (might be legacy or passed to an underlying library).
 * @param mT A double representing the transverse mass constraint (e.g., top quark mass).
 * @param mW A double representing the W boson mass constraint.
 * @param null A double value used to represent invalid or failed combinations/reconstructions.
 * @param perturb A double factor for potential kinematic perturbation.
 * @param steps A long integer specifying the number of iterations or steps for perturbation/optimization.
 *
 * @return A std::vector<std::pair<neutrino*, neutrino*>> containing pairs of pointers to reconstructed neutrino objects.
 *         Each pair represents a potential solution found by the combinatorial algorithm for an event. The structure
 *         implies that for each input event, multiple solutions (combinations) might be returned, or it might return
 *         the best solution per event. The caller is likely responsible for managing the memory of the pointed-to neutrino objects.
 */
std::vector<std::pair<neutrino*, neutrino*>> combinatorial(
    std::vector<double> met_,
    std::vector<double> phi_,
    std::vector<std::vector<particle_template*>> particles,
    std::string dev,
    double mT,
    double mW,
    double null,
    double perturb,
    long steps
);

} // namespace nusol
} // namespace pyc
