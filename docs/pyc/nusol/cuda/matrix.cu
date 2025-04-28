/**
 * @brief CUDA kernel designed for debugging the H-matrix computation process.
 *
 * This kernel calculates the H-matrix and its perpendicular counterpart (H_perp)
 * along with a suite of intermediate values useful for debugging and validation.
 * It operates on input tensors representing physical quantities like particle masses,
 * momenta, and transformation matrices. The computation is parameterized by a
 * floating-point type `scalar_t`.
 *
 * @tparam scalar_t The floating-point type (e.g., float, double) used for calculations.
 *                  This allows flexibility in precision requirements.
 *
 * @param masses A 2D tensor where each row contains mass hypotheses, typically
 *               including masses for the top quark, W boson, and neutrino (mT, mW, mNu).
 *               Shape: [num_mass_hypotheses, 3].
 * @param cosine A 2D tensor storing the cosine of the angle between the lepton and
 *               b-quark in the W boson rest frame, for each mass hypothesis.
 *               Shape: [num_events, num_mass_hypotheses].
 * @param rt A 3D tensor representing rotation/transformation matrices used to align
 *           coordinate systems, typically derived from particle momenta.
 *           Shape: [num_events, 3, 3].
 * @param pmc_l A 2D tensor containing the 4-momentum (E, Px, Py, Pz) of the lepton (e.g., muon)
 *              in the lab frame for each event. Shape: [num_events, 4].
 * @param m2l A 2D tensor storing the squared mass of the lepton for each event.
 *            Shape: [num_events, 1] or compatible.
 * @param b2l A 2D tensor storing the squared beta (v²/c²) of the lepton for each event.
 *            Shape: [num_events, 1] or compatible.
 * @param pmc_b A 2D tensor containing the 4-momentum (E, Px, Py, Pz) of the b-quark
 *              in the lab frame for each event. Shape: [num_events, 4].
 * @param m2b A 2D tensor storing the squared mass of the b-quark for each event.
 *            Shape: [num_events, 1] or compatible.
 * @param b2b A 2D tensor storing the squared beta (v²/c²) of the b-quark for each event.
 *            Shape: [num_events, 1] or compatible.
 * @param Hmatrix Output tensor where the computed H-matrices will be stored.
 *                The H-matrix is typically a 2x2 matrix related to the neutrino momentum solution.
 *                Shape: [num_events, num_mass_hypotheses, 4] (storing 2x2 matrix elements).
 * @param H_perp Output tensor where the computed perpendicular H-matrices will be stored.
 *               Shape: [num_events, num_mass_hypotheses, 4] (storing 2x2 matrix elements).
 * @param passed Output tensor (likely boolean or integer) indicating whether a valid solution
 *               was found for each event and mass hypothesis. Shape: [num_events * num_mass_hypotheses].
 * @param x0p Output tensor storing an intermediate calculation result (debug). Shape: [num_events * num_mass_hypotheses].
 * @param x0 Output tensor storing an intermediate calculation result (debug). Shape: [num_events * num_mass_hypotheses].
 * @param Sx Output tensor storing an intermediate calculation result (debug). Shape: [num_events * num_mass_hypotheses].
 * @param Sy Output tensor storing an intermediate calculation result (debug). Shape: [num_events * num_mass_hypotheses].
 * @param w Output tensor storing an intermediate calculation result (debug). Shape: [num_events * num_mass_hypotheses].
 * @param om2 Output tensor storing an intermediate calculation result (debug). Shape: [num_events * num_mass_hypotheses].
 * @param eps2 Output tensor storing an intermediate calculation result (debug). Shape: [num_events * num_mass_hypotheses].
 * @param x1 Output tensor storing an intermediate calculation result (debug). Shape: [num_events * num_mass_hypotheses].
 * @param y1 Output tensor storing an intermediate calculation result (debug). Shape: [num_events * num_mass_hypotheses].
 * @param z Output tensor storing an intermediate calculation result (debug). Shape: [num_events * num_mass_hypotheses].
 */
template <typename scalar_t>
__global__ void _hmatrix_debug(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> masses,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> cosine,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> rt,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_l,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2l,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2l,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_b,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2b,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2b,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Hmatrix,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H_perp,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> passed,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> x0p,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> x0,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Sx,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Sy,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> om2,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> eps2,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> x1,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> y1,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> z
);


/**
 * @brief Host function (part of the nusol_ namespace) that orchestrates the debug H-matrix calculation.
 *
 * This function serves as a high-level interface to the `_hmatrix_debug` CUDA kernel.
 * It takes tensors representing b-quark and muon momenta, along with mass hypotheses,
 * prepares the necessary inputs for the kernel (potentially calculating intermediate values like
 * rotation matrices, beta factors, etc.), launches the kernel, and returns the results,
 * including the H-matrix, H_perp, pass flags, and all the intermediate debug variables.
 *
 * @param pmc_b Pointer to a tensor containing the 4-momenta (E, Px, Py, Pz) of b-quarks
 *              for multiple events. Shape: [num_events, 4].
 * @param pmc_mu Pointer to a tensor containing the 4-momenta (E, Px, Py, Pz) of muons
 *               (or other leptons) for multiple events. Shape: [num_events, 4].
 * @param masses Pointer to a tensor containing mass hypotheses (mT, mW, mNu) to be tested.
 *               Shape: [num_mass_hypotheses, 3].
 * @return A std::map where keys are strings identifying the output tensors (e.g., "H",
 *         "H_perp", "passed", "x0", "Sx", etc.) and values are the corresponding
 *         torch::Tensor objects containing the results computed by the `_hmatrix_debug` kernel.
 */
std::map<std::string, torch::Tensor> nusol_::BaseDebug(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses);


/**
 * @brief CUDA kernel optimized for H-matrix computation using shared memory.
 *
 * This kernel calculates the H-matrix and its perpendicular counterpart (H_perp)
 * for multiple events and mass hypotheses. It is templated on `size_x` to allow
 * compile-time configuration of shared memory usage, potentially improving performance
 * by reducing global memory access, especially when `dx` (number of momentum entries)
 * is large. This version appears specialized for `double` precision.
 *
 * @tparam size_x A template parameter specifying the size of a dimension used for
 *                allocating shared memory. This likely corresponds to the number of
 *                events or a related dimension processed per block, influencing how
 *                data is partitioned and loaded into shared memory.
 *
 * @param masses A 2D tensor holding mass hypotheses (mT, mW, mNu).
 *               Shape: [num_mass_hypotheses, 3]. Assumed to be `double`.
 * @param cosine A 2D tensor storing the cosine values related to particle kinematics.
 *               Shape: [num_events, num_mass_hypotheses]. Assumed to be `double`.
 * @param rt A 3D tensor containing rotation/transformation matrices.
 *           Shape: [num_events, 3, 3]. Assumed to be `double`.
 * @param pmc_l A 2D tensor with lepton 4-momenta (E, Px, Py, Pz).
 *              Shape: [num_events, 4]. Assumed to be `double`.
 * @param m2l A 2D tensor with squared lepton masses.
 *            Shape: [num_events, 1] or compatible. Assumed to be `double`.
 * @param b2l A 2D tensor with squared lepton beta values (v²/c²).
 *            Shape: [num_events, 1] or compatible. Assumed to be `double`.
 * @param pmc_b A 2D tensor with b-quark 4-momenta (E, Px, Py, Pz).
 *              Shape: [num_events, 4]. Assumed to be `double`.
 * @param m2b A 2D tensor with squared b-quark masses.
 *            Shape: [num_events, 1] or compatible. Assumed to be `double`.
 * @param b2b A 2D tensor with squared b-quark beta values (v²/c²).
 *            Shape: [num_events, 1] or compatible. Assumed to be `double`.
 * @param Hmatrix Output tensor for the computed H-matrices.
 *                Shape: [num_events, num_mass_hypotheses, 4]. Will contain `double`.
 * @param H_perp Output tensor for the computed perpendicular H-matrices.
 *               Shape: [num_events, num_mass_hypotheses, 4]. Will contain `double`.
 * @param passed Output tensor indicating if a valid solution was found.
 *               Shape: [num_events * num_mass_hypotheses]. Will contain `double` (likely 0.0 or 1.0).
 * @param dy The number of mass points (hypotheses) being processed. Corresponds to `masses.size(0)`.
 * @param dx The number of momentum entries (events) being processed. Corresponds to `pmc_l.size(0)`.
 * @param sm A boolean flag indicating the computation mode (e.g., single mode vs. multiple modes),
 *           which might affect how calculations or checks are performed within the kernel.
 */
template <size_t size_x>
__global__ void _hmatrix(
    const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> masses,
    const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> cosine,
    const torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> rt,
    const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> pmc_l,
    const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> m2l,
    const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> b2l,
    const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> pmc_b,
    const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> m2b,
    const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> b2b,
    torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> Hmatrix,
    torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> H_perp,
    torch::PackedTensorAccessor64<double, 1, torch::RestrictPtrTraits> passed,
    const unsigned int dy,
    const unsigned int dx,
    const bool sm
);


/**
 * @brief Host function (part of the nusol_ namespace) for standard H-matrix calculation using tensor inputs.
 *
 * This function acts as an interface to the `_hmatrix` CUDA kernel (likely the shared memory version).
 * It takes tensors for b-quark and muon momenta, and a tensor for mass hypotheses.
 * It handles the setup, kernel launch (potentially choosing an appropriate `size_x` template
 * parameter based on hardware or input size), and returns the primary results: H-matrix,
 * H_perp, and the pass indicators. It assumes calculations are done in double precision.
 *
 * @param pmc_b Pointer to a tensor containing the 4-momenta (E, Px, Py, Pz) of b-quarks.
 *              Shape: [num_events, 4]. Expected to be `torch::kDouble`.
 * @param pmc_mu Pointer to a tensor containing the 4-momenta (E, Px, Py, Pz) of muons.
 *               Shape: [num_events, 4]. Expected to be `torch::kDouble`.
 * @param masses Pointer to a tensor containing mass hypotheses (mT, mW, mNu).
 *               Shape: [num_mass_hypotheses, 3]. Expected to be `torch::kDouble`.
 * @return A std::map containing the key output tensors:
 *         - "H": The computed H-matrix tensor. Shape: [num_events, num_mass_hypotheses, 4].
 *         - "H_perp": The computed perpendicular H-matrix tensor. Shape: [num_events, num_mass_hypotheses, 4].
 *         - "passed": Tensor indicating successful computation per event/hypothesis. Shape: [num_events * num_mass_hypotheses].
 *         All returned tensors will be of type `torch::kDouble`.
 */
std::map<std::string, torch::Tensor> nusol_::BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses);


/**
 * @brief CUDA kernel for H-matrix computation using a single, fixed set of masses.
 *
 * This kernel calculates the H-matrix and H_perp for multiple events but uses a single,
 * predefined set of masses (top quark, W boson, neutrino) passed as scalar arguments,
 * rather than reading them from a `masses` tensor. This is useful when analyzing data
 * assuming fixed, known particle masses. It is templated on the floating-point type `scalar_t`.
 *
 * @tparam scalar_t The floating-point type (e.g., float, double) for calculations.
 *
 * @param cosine A 2D tensor storing cosine values. Shape: [num_events, 1] or [num_events].
 * @param rt A 3D tensor containing rotation/transformation matrices. Shape: [num_events, 3, 3].
 * @param pmc_l A 2D tensor with lepton 4-momenta. Shape: [num_events, 4].
 * @param m2l A 2D tensor with squared lepton masses. Shape: [num_events, 1] or compatible.
 * @param b2l A 2D tensor with squared lepton beta values. Shape: [num_events, 1] or compatible.
 * @param pmc_b A 2D tensor with b-quark 4-momenta. Shape: [num_events, 4].
 * @param m2b A 2D tensor with squared b-quark masses. Shape: [num_events, 1] or compatible.
 * @param b2b A 2D tensor with squared b-quark beta values. Shape: [num_events, 1] or compatible.
 * @param Hmatrix Output tensor for the computed H-matrices. Since masses are fixed, the
 *                mass hypothesis dimension is removed. Shape: [num_events, 4].
 * @param H_perp Output tensor for the computed perpendicular H-matrices.
 *               Shape: [num_events, 4].
 * @param passed Output tensor indicating success for each event. Shape: [num_events].
 * @param mass_T The fixed mass of the top quark to use in calculations (passed as `double`,
 *               likely cast to `scalar_t` internally).
 * @param mass_W The fixed mass of the W boson to use in calculations (passed as `double`,
 *               likely cast to `scalar_t` internally).
 * @param mass_nu The fixed mass of the neutrino to use (defaults to 0). Passed as `double`,
 *                likely cast to `scalar_t` internally.
 */
template <typename scalar_t>
__global__ void _hmatrix(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> cosine,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> rt,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_l,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2l,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2l,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_b,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2b,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2b,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Hmatrix, // Note: Shape might be [num_events, 4] effectively
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H_perp,  // Note: Shape might be [num_events, 4] effectively
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> passed,
    double mass_T,
    double mass_W,
    double mass_nu = 0
);


/**
 * @brief Host function (part of the nusol_ namespace) for H-matrix calculation using manually specified masses.
 *
 * This function provides an interface to the `_hmatrix` CUDA kernel version that accepts
 * fixed masses as scalar arguments (`mass_T`, `mass_W`, `mass_nu`). It takes b-quark and
 * muon momentum tensors and the specific mass values, then launches the appropriate kernel.
 * This avoids the need to create a `masses` tensor when only one mass hypothesis is needed.
 * The precision (float or double) might be determined based on the input tensor types or
 * a default/template parameter in the implementation.
 *
 * @param pmc_b Pointer to a tensor containing the 4-momenta (E, Px, Py, Pz) of b-quarks.
 *              Shape: [num_events, 4].
 * @param pmc_mu Pointer to a tensor containing the 4-momenta (E, Px, Py, Pz) of muons.
 *               Shape: [num_events, 4].
 * @param mT The mass of the top quark (double precision) to be used for all events.
 * @param mW The mass of the W boson (double precision) to be used for all events.
 * @param mN The mass of the neutrino (double precision) to be used for all events.
 * @return A std::map containing the key output tensors:
 *         - "H": The computed H-matrix tensor. Shape: [num_events, 4].
 *         - "H_perp": The computed perpendicular H-matrix tensor. Shape: [num_events, 4].
 *         - "passed": Tensor indicating successful computation per event. Shape: [num_events].
 *         The data type of the returned tensors depends on the underlying kernel execution
 *         (likely matching the type of `pmc_b`/`pmc_mu` or potentially defaulting to double).
 */
std::map<std::string, torch::Tensor> nusol_::BaseMatrix(
    torch::Tensor* pmc_b,
    torch::Tensor* pmc_mu,
    double mT,
    double mW,
    double mN
);

