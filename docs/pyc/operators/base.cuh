/**
 * @brief Kernel to compute dot product.
 * @tparam scalar_t Numeric data type.
 * @param v1 First tensor.
 * @param v2 Second tensor.
 * @param out Output tensor.
 * @param dx Size in x-dimension.
 * @param dy Size in y-dimension.
 */
template <typename scalar_t>
__global__ void _dot(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v1, 
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v2, 
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
    const unsigned int dx, const unsigned int dy
);

/**
 * @brief Kernel to compute cross product.
 * @tparam scalar_t Numeric data type.
 * @param v1 First tensor.
 * @param v2 Second tensor.
 * @param out Output tensor.
 * @param dy Size in y-dimension.
 * @param dz Size in z-dimension.
 */
template <typename scalar_t>
__global__ void _cross(
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> v1, 
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v2, 
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> out, 
    const unsigned int dy, const unsigned int dz
);

/**
 * @brief Kernel to compute cosine or sine of angle between two vectors.
 * @tparam scalar_t Numeric data type.
 * @param x First tensor.
 * @param y Second tensor.
 * @param out Output tensor.
 * @param dx Size in x-dimension.
 * @param dy Size in y-dimension.
 * @param get_sin If true, compute sine instead of cosine.
 */
template <typename scalar_t>
__global__ void _costheta(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out,
    const unsigned int dx, const unsigned int dy, bool get_sin = false
);

/**
 * @brief Kernel to compute rotation in x-direction.
 * @tparam scalar_t Numeric data type.
 * @param angle Tensor of angles.
 * @param out Output tensor.
 * @param dx Size in x-dimension.
 */
template <typename scalar_t>
__global__ void _rx(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angle, 
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
    const unsigned int dx
);

/**
 * @brief Kernel to compute rotation in y-direction.
 * @tparam scalar_t Numeric data type.
 * @param angle Tensor of angles.
 * @param out Output tensor.
 * @param dx Size in x-dimension.
 */
template <typename scalar_t>
__global__ void _ry(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angle, 
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
    const unsigned int dx
);

/**
 * @brief Kernel to compute rotation in z-direction.
 * @tparam scalar_t Numeric data type.
 * @param angle Tensor of angles.
 * @param out Output tensor.
 * @param dx Size in x-dimension.
 */
template <typename scalar_t>
__global__ void _rz(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angle, 
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
    const unsigned int dx
);

/**
 * @brief Kernel to handle combined rotations and transformations.
 * @tparam scalar_t Numeric data type.
 * @param pmc Input data tensor.
 * @param phi Angles in phi dimension.
 * @param theta Angles in theta dimension.
 * @param out Output tensor.
 */
template <typename scalar_t>
__global__ void _rt(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> theta, 
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out
);

/**
 * @brief Kernel to compute cofactors of a 3x3 matrix.
 * @tparam scalar_t Numeric data type.
 * @tparam size_x Shared memory row size.
 * @param matrix Input tensor.
 * @param out Output tensor of cofactors.
 */
template <typename scalar_t, size_t size_x>
__global__ void _cofactor(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out
);

/**
 * @brief Kernel to compute determinants of 3x3 matrices.
 * @tparam scalar_t Numeric data type.
 * @tparam size_x Shared memory row size.
 * @param matrix Input tensor.
 * @param out Output tensor of determinants.
 */
template <typename scalar_t, size_t size_x>
__global__ void _determinant(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out
);

/**
 * @brief Kernel to compute inverse of 3x3 matrices.
 * @tparam scalar_t Numeric data type.
 * @tparam size_x Shared memory row size.
 * @param matrix Input tensor.
 * @param inv Output tensor for inverted matrices.
 * @param det Output tensor for determinants.
 */
template <typename scalar_t, size_t size_x>
__global__ void _inverse(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> inv,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> det
);

/**
 * @brief Kernel to compute eigenvalues of 3x3 matrices.
 * @tparam scalar_t Numeric data type.
 * @tparam size_x Shared memory row size.
 * @param matrix Input tensor.
 * @param real Real parts of eigenvalues.
 * @param img Imag imaginary parts of eigenvalues.
 */
template <typename scalar_t, size_t size_x>
__global__ void _eigenvalue(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> real,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> img
);
