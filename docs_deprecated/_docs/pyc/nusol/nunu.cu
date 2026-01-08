/**
 * @brief Initializes intermediate tensors for the double neutrino solutions algorithm.
 * @details This CUDA kernel performs the initial setup for the NuNu algorithm. It operates on batches of events, 
 *          processing each event in parallel using CUDA threads. The primary tasks include pre-computing 
 *          intermediate values derived from the input base matrices (H1, H2), their inverses (H1_inv, H2_inv), 
 *          and the missing transverse momentum (met_xy). These pre-computed values are stored in the output 
 *          tensors (n, n_, N, K, K_, S) and are crucial for the subsequent steps of the NuNu algorithm. 
 *          Shared memory is utilized within the kernel (indicated by the size_x template parameter) to optimize 
 *          memory access patterns and improve performance by facilitating data sharing among threads within a block.
 *
 * @tparam scalar_t The floating-point data type used for calculations (e.g., float, double). This allows for 
 *                  flexible precision depending on the requirements.
 * @tparam size_x   Specifies the size of the dimension used for shared memory allocation within the kernel. 
 *                  This parameter influences performance and resource usage.
 * 
 * @param met_xy [in] A 2D tensor representing the missing transverse momentum (MET) for each event. 
 *                    Shape: (batch_size, 2), where the columns correspond to the x and y components of MET.
 * @param H1_inv [in] A 3D tensor containing the pre-calculated inverse of the first base matrix (H1) for each event.
 *                    Shape: (batch_size, 2, 2).
 * @param H2_inv [in] A 3D tensor containing the pre-calculated inverse of the second base matrix (H2) for each event.
 *                    Shape: (batch_size, 2, 2).
 * @param H1 [in]     A 3D tensor representing the first base matrix for each event.
 *                    Shape: (batch_size, 2, 2).
 * @param H2 [in]     A 3D tensor representing the second base matrix for each event.
 *                    Shape: (batch_size, 2, 2).
 * @param n [out]     A 3D tensor to store intermediate results calculated during initialization. 
 *                    Shape: (batch_size, 2, 1). This typically holds components related to the first neutrino solution.
 * @param n_ [out]    A 3D tensor to store intermediate results calculated during initialization.
 *                    Shape: (batch_size, 2, 1). This typically holds components related to the second neutrino solution.
 * @param N [out]     A 3D tensor to store intermediate results, often related to the normal vectors or combined properties.
 *                    Shape: (batch_size, 2, 2).
 * @param K [out]     A 3D tensor storing intermediate results, potentially related to kinematic constraints or transformations for the first solution.
 *                    Shape: (batch_size, 2, 2).
 * @param K_ [out]    A 3D tensor storing intermediate results, potentially related to kinematic constraints or transformations for the second solution.
 *                    Shape: (batch_size, 2, 2).
 * @param S [out]     A 3D tensor storing intermediate results, often representing a sum or combination required for vertex calculations.
 *                    Shape: (batch_size, 2, 2).
 */
template <typename scalar_t, size_t size_x>
__global__ void _nunu_init_(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> met_xy,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H1_inv, 
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H2_inv,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H1, 
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H2,

          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> n ,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> n_,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> N ,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K ,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K_,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> S
);

/**
 * @brief Computes the vertex points for the double neutrino solutions in the NuNu algorithm.
 * @details This CUDA kernel calculates the potential vertex solutions for the two neutrinos based on the 
 *          intermediate tensors S, K, and K_ computed in the initialization phase (`_nunu_init_`). 
 *          It determines the intersection points or closest approach points related to the kinematic constraints. 
 *          The kernel updates several output tensors: `n` and `n_` might be refined, while `v`, `v_` store the 
 *          vertex components, `ds` stores a distance or discriminant value related to the solutions, and `nu0`, `nu1` 
 *          store the initial estimates or components of the two neutrino momenta. Like `_nunu_init_`, this kernel 
 *          operates on batches of events and leverages shared memory (controlled by `size_x`) for efficient 
 *          computation and data access within thread blocks.
 *
 * @tparam scalar_t The floating-point data type used for calculations (e.g., float, double).
 * @tparam size_x   Specifies the size of the dimension used for shared memory allocation within the kernel.
 * 
 * @param S [in]   Input tensor containing intermediate results from `_nunu_init_`. 
 *                 Shape: (batch_size, 2, 2). Represents combined kinematic information.
 * @param K [in]   Input tensor containing intermediate results from `_nunu_init_`.
 *                 Shape: (batch_size, 2, 2). Related to constraints for the first neutrino.
 * @param K_ [in]  Input tensor containing intermediate results from `_nunu_init_`.
 *                 Shape: (batch_size, 2, 2). Related to constraints for the second neutrino.
 * @param n [in, out] Input/Output tensor, potentially refined during vertex calculation.
 *                 Shape: (batch_size, 2, 1). Related to the first neutrino solution components.
 * @param n_ [in, out] Input/Output tensor, potentially refined during vertex calculation.
 *                 Shape: (batch_size, 2, 1). Related to the second neutrino solution components.
 * @param v [out]  Output tensor storing components of the first vertex solution.
 *                 Shape: (batch_size, 2, 1).
 * @param v_ [out] Output tensor storing components of the second vertex solution.
 *                 Shape: (batch_size, 2, 1).
 * @param ds [out] Output tensor storing a discriminant or distance value associated with the vertex solutions for each event.
 *                 Shape: (batch_size, 1). Can indicate the quality or existence of real solutions.
 * @param nu0 [out] Output tensor storing components or initial estimates of the first neutrino's momentum.
 *                  Shape: (batch_size, 2, 1).
 * @param nu1 [out] Output tensor storing components or initial estimates of the second neutrino's momentum.
 *                  Shape: (batch_size, 2, 1).
 */
template <typename scalar_t, size_t size_x>
__global__ void _nunu_vp_(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> S, 
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K, 
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K_,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> n,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> n_,

          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v_,
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> ds,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu0,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu1
);

/**
 * @brief Iteratively computes the residual and refines neutrino solutions for the NuNu algorithm.
 * @details This CUDA kernel implements an iterative refinement process for the neutrino momentum solutions. 
 *          It calculates a residual based on the difference between the calculated transverse momentum (derived 
 *          from the current neutrino estimates `nu1`, `nu2` and the perpendicular components of the base matrices 
 *          `H_perp`, `H_perp_`) and the measured missing transverse momentum (`metxy`). The kernel iteratively 
 *          updates the neutrino momentum estimates (`nu1`, `nu2`) using a gradient descent-like approach with a 
 *          given `step` size, aiming to minimize this residual. The process continues until the residual falls 
 *          below a specified `tol` (tolerance) or a maximum number of iterations (`timeout`) is reached. 
 *          The final residual magnitude is stored in `dst`. Shared memory (controlled by `size_x` and `size_y`) 
 *          is used to enhance performance during the iterative calculations.
 *
 * @tparam scalar_t The floating-point data type used for calculations (e.g., float, double).
 * @tparam size_x   Specifies the size of the first dimension for shared memory allocation.
 * @tparam size_y   Specifies the size of the second dimension for shared memory allocation.
 * 
 * @param metxy [in] Input tensor containing the measured missing transverse momentum (MET) for each event.
 *                   Shape: (batch_size, 2).
 * @param H_perp [in] Input tensor representing the perpendicular component of the first base matrix.
 *                    Shape: (batch_size, 2, 1). Used to project neutrino momentum.
 * @param H_perp_ [in] Input tensor representing the perpendicular component of the second base matrix.
 *                     Shape: (batch_size, 2, 1). Used to project neutrino momentum.
 * @param nu1 [in, out] Input/Output tensor representing the momentum estimate for the first neutrino. It is iteratively updated.
 *                      Shape: (batch_size, 2, 1).
 * @param nu2 [in, out] Input/Output tensor representing the momentum estimate for the second neutrino. It is iteratively updated.
 *                      Shape: (batch_size, 2, 1).
 * @param dst [out] Output tensor storing the final residual magnitude (distance) after the iterative refinement for each event.
 *                  Shape: (batch_size, 1).
 * @param tol [in] The convergence tolerance. The iteration stops if the residual magnitude drops below this value.
 * @param step [in] The step size used in the iterative update rule (e.g., learning rate in gradient descent).
 * @param timeout [in] The maximum number of iterations allowed before the refinement process stops, preventing infinite loops.
 */
template <typename scalar_t, size_t size_x, size_t size_y>
__global__ void _residual_(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> metxy, 
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H_perp, 
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H_perp_,
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu1, 
          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu2, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> dst, 
          const double tol, const double step, const unsigned int timeout
);

/**
 * @brief Executes the core NuNu algorithm using pre-computed base matrices.
 * @details This function serves as the main entry point for the NuNu algorithm when the base matrices (H1, H2) 
 *          and their perpendicular components (H1_perp, H2_perp) are already computed. It orchestrates the 
 *          entire process:
 *          1. Allocates necessary intermediate and output tensors on the GPU.
 *          2. Launches the `_nunu_init_` kernel to pre-compute initial values.
 *          3. Launches the `_nunu_vp_` kernel to calculate vertex solutions and initial neutrino estimates.
 *          4. Launches the `_residual_` kernel to iteratively refine the neutrino solutions based on MET consistency.
 *          5. Copies the results back from the GPU to CPU tensors.
 *          6. Packages the final neutrino momentum solutions (`nu1`, `nu2`), vertex information (`v`, `v_`), 
 *             discriminant (`ds`), and residual (`dst`) into a map.
 *
 * @param H1_ [in] Pointer to a torch::Tensor representing the first base matrix H1. Shape: (batch_size, 2, 2).
 * @param H1_perp [in] Pointer to a torch::Tensor representing the perpendicular component of H1. Shape: (batch_size, 2, 1).
 * @param H2_ [in] Pointer to a torch::Tensor representing the second base matrix H2. Shape: (batch_size, 2, 2).
 * @param H2_perp [in] Pointer to a torch::Tensor representing the perpendicular component of H2. Shape: (batch_size, 2, 1).
 * @param met_xy [in] Pointer to a torch::Tensor representing the measured missing transverse momentum (MET). Shape: (batch_size, 2).
 * @param null [in] A numerical value used potentially for masking or indicating invalid results (although its specific use isn't detailed here, it might be related to initialization or output masking).
 * @param step [in] The step size parameter passed to the `_residual_` kernel for the iterative refinement.
 * @param tolerance [in] The convergence tolerance parameter passed to the `_residual_` kernel.
 * @param timeout [in] The maximum number of iterations parameter passed to the `_residual_` kernel.
 * 
 * @return A std::map<std::string, torch::Tensor> containing the results. The map typically includes keys like 
 *         "nu1", "nu2" (final neutrino momenta), "v", "v_" (vertex solutions), "ds" (discriminant), and "dst" (residual). 
 *         The tensors hold the results for the entire batch.
 */
std::map<std::string, torch::Tensor> nusol_::NuNu(
    torch::Tensor* H1_, torch::Tensor* H1_perp, torch::Tensor* H2_, torch::Tensor* H2_perp, torch::Tensor* met_xy,
    double null, const double step, const double tolerance, const unsigned int timeout
);

/**
 * @brief Executes the NuNu algorithm using Particle Momentum Components (PMC) and event-wise masses.
 * @details This overload of the NuNu function provides an interface where the inputs are given in terms of 
 *          particle momentum components (PMCs) for the involved particles (b-quarks: b1, b2; leptons: mu1, mu2) 
 *          and event-wise masses for the parent particles (e.g., top quarks or W bosons, m1, m2). 
 *          It performs the following steps:
 *          1. Constructs the base matrices (H1, H2) and their perpendicular components (H1_perp, H2_perp) 
 *             internally using the provided PMC inputs and mass tensors (m1, m2). This involves applying the 
 *             kinematic constraints derived from the assumed decay topology (e.g., t -> b W, W -> l nu).
 *          2. Calls the core NuNu implementation (`nusol_::NuNu(H1_, H1_perp, ..., timeout)`) with the 
 *             constructed matrices and other parameters.
 *          3. Returns the resulting map containing the calculated neutrino solutions and related quantities.
 *          This function simplifies the usage when the primary inputs are particle four-vectors rather than 
 *          pre-computed matrices.
 *
 * @param pmc_b1 [in] Pointer to a torch::Tensor containing the four-momentum components (e.g., Px, Py, Pz, E or Pt, Eta, Phi, E) 
 *                   of the first b-quark for each event. Shape depends on the convention (e.g., batch_size, 4).
 * @param pmc_b2 [in] Pointer to a torch::Tensor containing the four-momentum components of the second b-quark.
 * @param pmc_mu1 [in] Pointer to a torch::Tensor containing the four-momentum components of the first lepton (muon/electron).
 * @param pmc_mu2 [in] Pointer to a torch::Tensor containing the four-momentum components of the second lepton.
 * @param met_xy [in] Pointer to a torch::Tensor representing the measured missing transverse momentum (MET). Shape: (batch_size, 2).
 * @param null [in] A numerical value used potentially for masking or indicating invalid results.
 * @param m1 [in] Pointer to a torch::Tensor containing the mass of the first parent particle (e.g., top quark or W boson) for each event. Shape: (batch_size, 1).
 * @param m2 [in] Pointer to a torch::Tensor containing the mass of the second parent particle for each event. Shape: (batch_size, 1).
 * @param step [in] The step size parameter for the iterative refinement in the core NuNu algorithm.
 * @param tolerance [in] The convergence tolerance parameter for the iterative refinement.
 * @param timeout [in] The maximum number of iterations parameter for the iterative refinement.
 * 
 * @return A std::map<std::string, torch::Tensor> containing the results ("nu1", "nu2", "v", "v_", "ds", "dst").
 */
std::map<std::string, torch::Tensor> nusol_::NuNu(
        torch::Tensor* pmc_b1,  torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
        torch::Tensor* met_xy,  double null, torch::Tensor* m1, torch::Tensor* m2, 
        const double step, const double tolerance, unsigned int timeout
);

/**
 * @brief Executes the NuNu algorithm using PMC inputs and fixed mass parameters.
 * @details This overload provides a simplified interface for the NuNu algorithm when fixed, known masses can be 
 *          assumed for the parent particles (e.g., top quarks massT1, massT2 and W boson masses massW1, massW2) 
 *          across all events in the batch. 
 *          It functions similarly to the previous overload:
 *          1. Constructs the base matrices (H1, H2) and their perpendicular components using the provided PMC inputs 
 *             (pmc_b1, pmc_b2, pmc_mu1, pmc_mu2) and the fixed mass values (massT1, massW1, massT2, massW2). 
 *             The kinematic constraints are derived assuming these fixed masses.
 *          2. Calls the core NuNu implementation (`nusol_::NuNu(H1_, H1_perp, ..., timeout)`) with the constructed 
 *             matrices, MET, and default refinement parameters (step=1e-12, tolerance=1e-12, timeout=100). 
 *             Note: This overload uses fixed default values for step, tolerance, and timeout.
 *          3. Returns the resulting map containing the calculated neutrino solutions.
 *          This is convenient when analyzing events where standard model masses can be reliably assumed.
 *
 * @param pmc_b1 [in] Pointer to a torch::Tensor containing the four-momentum components of the first b-quark.
 * @param pmc_b2 [in] Pointer to a torch::Tensor containing the four-momentum components of the second b-quark.
 * @param pmc_mu1 [in] Pointer to a torch::Tensor containing the four-momentum components of the first lepton.
 * @param pmc_mu2 [in] Pointer to a torch::Tensor containing the four-momentum components of the second lepton.
 * @param met_xy [in] Pointer to a torch::Tensor representing the measured missing transverse momentum (MET). Shape: (batch_size, 2).
 * @param null [in] A numerical value used potentially for masking or indicating invalid results.
 * @param massT1 [in] The fixed mass assumed for the first parent top quark (or similar particle).
 * @param massW1 [in] The fixed mass assumed for the first intermediate W boson (or similar particle).
 * @param massT2 [in] The fixed mass assumed for the second parent top quark.
 * @param massW2 [in] The fixed mass assumed for the second intermediate W boson.
 * 
 * @return A std::map<std::string, torch::Tensor> containing the results ("nu1", "nu2", "v", "v_", "ds", "dst").
 */
std::map<std::string, torch::Tensor> nusol_::NuNu(
        torch::Tensor* pmc_b1,  torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
        torch::Tensor* met_xy, double null, double massT1, double massW1, double massT2, double massW2
);

