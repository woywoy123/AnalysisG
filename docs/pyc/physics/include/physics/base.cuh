/**
 * @brief Kernel function computing p² from momentum components.
 * @tparam scalar_t Floating-point type.
 * @param pmc Momentum tensor accessor.
 * @param p2 Output tensor accessor for squared momentum.
 * @param dy Number of shared data elements in y-dimension.
 */
template <typename scalar_t> 
__global__ void _P2K(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> p2,
    const unsigned int dy
);

/**
 * @brief Kernel function computing p from momentum components.
 * @tparam scalar_t Floating-point type.
 * @param pmc Momentum tensor accessor.
 * @param p Output tensor accessor for momentum magnitude.
 * @param dy Number of shared data elements in y-dimension.
 */
template <typename scalar_t> 
__global__ void _PK(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> p,
    const unsigned int dy
);

/**
 * @brief Kernel function calculating β².
 * @tparam scalar_t Floating-point type.
 * @param pmc Momentum tensor accessor.
 * @param b2 Output tensor accessor for β².
 */
template <typename scalar_t>
__global__ void _Beta2(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2
);

/**
 * @brief Kernel function calculating β.
 * @tparam scalar_t Floating-point type.
 * @param pmc Momentum tensor accessor.
 * @param b Output tensor accessor for β.
 */
template <typename scalar_t>
__global__ void _Beta(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b
);

/**
 * @brief Kernel function calculating invariant mass squared.
 * @tparam scalar_t Floating-point type.
 * @param pmc Momentum tensor accessor.
 * @param m2 Output tensor accessor for mass squared.
 */
template <typename scalar_t>
__global__ void _M2(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2
);

/**
 * @brief Kernel function calculating invariant mass.
 * @tparam scalar_t Floating-point type.
 * @param pmc Momentum tensor accessor.
 * @param m Output tensor accessor for mass.
 */
template <typename scalar_t>
__global__ void _M(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m
);

/**
 * @brief Kernel function calculating transverse mass squared.
 * @tparam scalar_t Floating-point type.
 * @param pmc Momentum tensor accessor.
 * @param mt2 Output tensor accessor for transverse mass squared.
 */
template <typename scalar_t>
__global__ void _Mt2(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mt2
);

/**
 * @brief Kernel function calculating transverse mass.
 * @tparam scalar_t Floating-point type.
 * @param pmc Momentum tensor accessor.
 * @param mt Output tensor accessor for transverse mass.
 */
template <typename scalar_t>
__global__ void _Mt(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mt
);

/**
 * @brief Kernel function computing the angle θ from momentum components.
 * @tparam scalar_t Floating-point type.
 * @param pmc Momentum tensor accessor.
 * @param theta Output tensor accessor for θ.
 */
template <typename scalar_t> 
__global__ void _theta(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> theta
);

/**
 * @brief Kernel function calculating ΔR between two 4-vectors.
 * @tparam scalar_t Floating-point type.
 * @param pmu1 First 4-momentum tensor accessor.
 * @param pmu2 Second 4-momentum tensor accessor.
 * @param dr Output tensor accessor for ΔR.
 */
template <typename scalar_t> 
__global__ void _deltar(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu1, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu2, 
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> dr
);
