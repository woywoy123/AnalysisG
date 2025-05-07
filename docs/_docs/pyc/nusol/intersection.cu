/**
 * @brief CUDA kernel designed to conditionally swap two batches of matrices (A and B) based on their determinants and subsequently compute the product of the inverse of the first matrix (potentially swapped A) and the second matrix (potentially swapped B).
 *
 * @details This kernel operates on batches of 2x2 matrices represented by the input tensors A and B.
 * For each corresponding pair of matrices A_i and B_i in the batch:
 * 1. It compares the absolute values of their determinants, detA_i and detB_i.
 * 2. If |detA_i| < |detB_i|, the matrices A_i and B_i are swapped conceptually for the subsequent calculation. This step aims to improve numerical stability by ensuring the matrix to be inverted has a larger determinant (further from singularity).
 * 3. It computes the inverse of the matrix designated as 'A' after the potential swap (let's call it A'_i).
 * 4. It calculates the matrix product A'_i⁻¹ * B'_i, where B'_i is the matrix designated as 'B' after the potential swap.
 * 5. The resulting 2x2 matrix is stored in the corresponding location in the output tensor `inv_A_dot_B`.
 *
 * The computation is parallelized across the batches using CUDA's execution model. The template parameter `size_x` likely influences the thread block dimensions and potentially shared memory usage for optimizing the matrix inversion and multiplication steps.
 *
 * @tparam size_x Specifies the dimension of the thread block along the X-axis. This parameter is used to configure the kernel launch and potentially optimize resource usage (e.g., shared memory allocation) based on the hardware architecture.
 *
 * @param inv_A_dot_B Output tensor accessor. This is a 3D tensor (batch_size x 2 x 2) where the results of the computation (A'⁻¹ * B') for each item in the batch will be stored. It uses `double` precision floating-point numbers.
 * @param A Input tensor accessor for the first batch of matrices. This is a 3D tensor (batch_size x 2 x 2) containing the matrices A_i. It uses `double` precision.
 * @param B Input tensor accessor for the second batch of matrices. This is a 3D tensor (batch_size x 2 x 2) containing the matrices B_i. It uses `double` precision.
 * @param detA Input tensor accessor containing the pre-computed determinants for each matrix in A. This is a 2D tensor (batch_size x 1) storing det(A_i). It uses `double` precision.
 * @param detB Input tensor accessor containing the pre-computed determinants for each matrix in B. This is a 2D tensor (batch_size x 1) storing det(B_i). It uses `double` precision.
 */
template <size_t size_x>
__global__ void _swapAB(
    torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> inv_A_dot_B,
    torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> detA,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> detB
);

/**
 * @brief CUDA kernel to compute factors related to degenerate cases arising from the generalized eigenvalue problem (Ax = lambda*Bx) or similar matrix pencil operations.
 *
 * @details This kernel analyzes the eigenvalues (provided as separate real and imaginary parts) and the original matrices A and B to identify and characterize degenerate situations. Degeneracy might occur, for example, when eigenvalues are infinite or indeterminate, or when eigenvectors correspond to null spaces.
 * The kernel computes specific factors or representations (stored in `Lins`) that describe these degeneracies, potentially representing lines at infinity, common null vectors, or other geometric interpretations depending on the context where A and B originate (e.g., conic sections, quadric surfaces).
 * The `nulls` parameter serves as a tolerance threshold to handle near-zero values during calculations involving determinants, eigenvalue comparisons, or vector normalizations, preventing division by zero and identifying numerically degenerate cases.
 *
 * @tparam scalar_t The data type used for computations (e.g., `float`, `double`). This allows the kernel to be flexible with different precision requirements.
 * @tparam size_x Specifies the dimension of the thread block along the X-axis, influencing kernel launch configuration and potential optimizations.
 *
 * @param real Input tensor accessor holding the real parts of the computed eigenvalues for each matrix pair (A_i, B_i). This is expected to be a 2D tensor (batch_size x num_eigenvalues).
 * @param imag Input tensor accessor holding the imaginary parts of the computed eigenvalues. This corresponds to the `real` tensor and has the same dimensions.
 * @param A Input tensor accessor for the first batch of matrices (e.g., 2x2 or 3x3). This is a 3D tensor (batch_size x dim x dim).
 * @param B Input tensor accessor for the second batch of matrices, corresponding to A. This is a 3D tensor (batch_size x dim x dim).
 * @param Lins Output tensor accessor. This 4D tensor (batch_size x num_degenerate_cases x num_params x dim) stores the computed factors or geometric representations describing the degenerate cases identified for each input pair. The exact meaning of the dimensions depends on the specific type of degeneracy being characterized.
 * @param nulls A `double` precision floating-point value representing the tolerance threshold. Values smaller than `nulls` in magnitude might be treated as zero during the analysis of degenerate conditions.
 */
template <typename scalar_t, size_t size_x>
__global__ void _factor_degen(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> real,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> imag,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> Lins,
    const double nulls
);

/**
 * @brief CUDA kernel to compute the intersection points between batches of lines and ellipses.
 *
 * @details This kernel takes descriptions of ellipses and lines as input and calculates their points of intersection. Ellipses are likely represented by their quadratic form coefficients, and lines by parametric or implicit equations.
 * The kernel utilizes intermediate complex number calculations (stored in `real`) potentially arising from solving the system of equations defining the line and the ellipse. This often involves finding roots of a quadratic or quartic polynomial.
 * For each line-ellipse pair in the batch, it computes up to two intersection points (for a line and an ellipse). The coordinates of these points are stored in `s_pts`.
 * Additionally, it calculates associated distances (`s_dst`), which could represent the distance of the intersection points from the start of the line segment, the distance from the ellipse center, or another relevant metric depending on the application.
 * The `nulls` parameter is used as a tolerance for floating-point comparisons, helping to manage numerical precision issues, identify tangent intersections, or handle cases where no real intersection exists within the tolerance.
 *
 * @tparam scalar_t The data type used for computations, expected to be `double` based on the parameter types, ensuring high precision for geometric calculations.
 * @tparam size_x Specifies the dimension of the thread block along the X-axis for CUDA kernel launch configuration.
 *
 * @param real Input/Output tensor accessor holding complex intermediate values. This 4D tensor (batch_size x num_lines x num_ellipses x intermediate_data_size) likely stores roots of polynomials or other complex results generated during the intersection calculation. It uses `c10::complex<double>`.
 * @param ellipse Input tensor accessor describing the batch of ellipses. This 3D tensor (batch_size x num_ellipses x ellipse_params) contains the parameters defining each ellipse (e.g., coefficients of the quadratic equation Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0). It uses `double` precision.
 * @param lines Input tensor accessor describing the batch of lines. This 4D tensor (batch_size x num_lines x line_params x dim) contains the parameters defining each line (e.g., start/end points, or coefficients of the implicit form ax + by + c = 0). It uses `double` precision.
 * @param s_pts Output tensor accessor storing the coordinates of the intersection points. This 3D tensor (batch_size x num_intersections x point_dim) will contain the (x, y) coordinates of each found intersection. It uses `double` precision.
 * @param s_dst Output tensor accessor storing distances associated with the intersection points. This 2D tensor (batch_size x num_intersections) holds a distance metric for each point in `s_pts`. It uses `double` precision.
 * @param nulls A `double` precision floating-point value used as a tolerance threshold for numerical comparisons during the intersection calculation (e.g., checking if a discriminant is close to zero).
 */
template <typename scalar_t, size_t size_x>
__global__ void _intersections(
    torch::PackedTensorAccessor64<c10::complex<double>, 4, torch::RestrictPtrTraits> real,
    torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> ellipse,
    torch::PackedTensorAccessor64<double, 4, torch::RestrictPtrTraits> lines,
    torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> s_pts,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> s_dst,
    const double nulls
);

/**
 * @brief Host function that computes the intersection points between geometric entities represented by matrix pencils (A, B).
 *
 * @details This function serves as the main entry point for an intersection algorithm, likely involving conic sections or quadric surfaces defined by pairs of symmetric matrices (A, B). The intersection is found by analyzing the properties of the matrix pencil A - lambda*B.
 *
 * The process typically involves:
 * 1. Preprocessing: Calculating determinants (potentially using `_swapAB`'s logic implicitly or explicitly) to handle numerical stability.
 * 2. Eigenvalue Problem: Solving the generalized eigenvalue problem associated with the matrix pencil (A, B) to find eigenvalues and eigenvectors. This step might involve calls to specialized linear algebra routines (e.g., LAPACK, cuSOLVER).
 * 3. Degeneracy Handling: Identifying and characterizing degenerate cases (e.g., infinite eigenvalues, common null spaces) using kernels like `_factor_degen`. These degeneracies often correspond to specific geometric configurations (e.g., lines at infinity, shared components).
 * 4. Intersection Calculation: Parameterizing the intersection curve or points based on the eigenvalues/eigenvectors and potentially calling kernels like `_intersections` if the problem reduces to line-conic intersections after parameterization or projection.
 * 5. Postprocessing: Organizing the computed intersection points (`s_pts`) and associated data (e.g., distances `s_dst`) into output tensors.
 *
 * The `nulls` parameter is propagated to the underlying CUDA kernels to ensure consistent handling of numerical tolerances throughout the computation.
 *
 * @param A Pointer to a PyTorch tensor representing the first set of matrices (e.g., defining the first set of quadrics/conics). Expected shape might be (batch_size x dim x dim).
 * @param B Pointer to a PyTorch tensor representing the second set of matrices (e.g., defining the second set of quadrics/conics). Must correspond to tensor A. Expected shape might be (batch_size x dim x dim).
 * @param nulls A `double` precision tolerance value used for numerical stability checks (e.g., identifying near-zero determinants or eigenvalues) within the algorithm and its constituent kernel calls.
 *
 * @return A `std::map<std::string, torch::Tensor>` containing the results. The map typically includes:
 *         - Key "solutions": A PyTorch tensor holding the coordinates of the computed intersection points.
 *         - Key "distances": A PyTorch tensor holding associated distance values or parameters for each intersection point.
 *         Other keys might be included depending on the specific details of the intersection algorithm (e.g., information about degenerate components).
 */
std::map<std::string, torch::Tensor> nusol_::Intersection(
    torch::Tensor* A,
    torch::Tensor* B,
    double nulls
);