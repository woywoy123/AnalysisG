/**
 * @file base.cuh
 * @brief This file defines CUDA kernels for transforming particle kinematics between different coordinate systems.
 *
 * The kernels primarily operate on particle 4-momentum representations. Two common representations are used:
 * 1.  `(pt, eta, phi, E)`: Transverse momentum, pseudorapidity, azimuthal angle, and energy. Often referred to as `pmu` in variable names.
 * 2.  `(px, py, pz, E)`: Cartesian momentum components (x, y, z) and energy. Often referred to as `pmc` in variable names.
 *
 * Input and output data are passed using `torch::PackedTensorAccessor64`, a PyTorch C++ API structure for accessing tensor data on the GPU.
 * Kernels are templated on the scalar type (`scalar_t`) to support different floating-point precisions (e.g., `float`, `double`).
 * Some kernels utilize shared memory for intermediate calculations to optimize memory access patterns and improve performance.
 * These kernels assume input tensors are structured such that each row represents a particle and columns represent the kinematic variables.
 */

/**
 * @brief CUDA kernel to calculate the Cartesian x-component of momentum (px).
 *
 * Calculates `px = pt * cos(phi)` for each particle.
 *
 * @tparam scalar_t The data type (e.g., float, double) of the tensor elements.
 * @param pt A PackedTensorAccessor64 representing the transverse momentum (`pt`) for each particle. Assumed shape: `[n_particles, 1]`.
 * @param phi A PackedTensorAccessor64 representing the azimuthal angle (`phi`) for each particle. Assumed shape: `[n_particles, 1]`.
 * @param px A PackedTensorAccessor64 to store the calculated x-component of momentum (`px`). Assumed shape: `[n_particles, 1]`.
 */

/**
 * @brief CUDA kernel to calculate the Cartesian y-component of momentum (py).
 *
 * Calculates `py = pt * sin(phi)` for each particle.
 *
 * @tparam scalar_t The data type (e.g., float, double) of the tensor elements.
 * @param pt A PackedTensorAccessor64 representing the transverse momentum (`pt`) for each particle. Assumed shape: `[n_particles, 1]`.
 * @param phi A PackedTensorAccessor64 representing the azimuthal angle (`phi`) for each particle. Assumed shape: `[n_particles, 1]`.
 * @param py A PackedTensorAccessor64 to store the calculated y-component of momentum (`py`). Assumed shape: `[n_particles, 1]`.
 */

/**
 * @brief CUDA kernel to calculate the Cartesian z-component of momentum (pz).
 *
 * Calculates `pz = pt * sinh(eta)` for each particle.
 *
 * @tparam scalar_t The data type (e.g., float, double) of the tensor elements.
 * @param pt A PackedTensorAccessor64 representing the transverse momentum (`pt`) for each particle. Assumed shape: `[n_particles, 1]`.
 * @param eta A PackedTensorAccessor64 representing the pseudorapidity (`eta`) for each particle. Assumed shape: `[n_particles, 1]`.
 * @param pz A PackedTensorAccessor64 to store the calculated z-component of momentum (`pz`). Assumed shape: `[n_particles, 1]`.
 */

/**
 * @brief CUDA kernel to transform 4-momentum from (pt, eta, phi, E/Mass) to (px, py, pz, E/Mass).
 *
 * This kernel converts the first three components of a 4-vector from cylindrical-like coordinates
 * (pt, eta, phi) to Cartesian coordinates (px, py, pz). The fourth component (Energy or Mass)
 * is typically passed through unchanged.
 * It uses shared memory (`pmx`) to cache the input `pmu` components for a thread block,
 * potentially improving memory access efficiency. Each thread calculates one output component.
 * Requires the kernel launch configuration to have `blockDim.y` equal to the number of components (typically 4).
 * Relies on `px_`, `py_`, `pz_` device functions for the component calculations.
 *
 * @tparam scalar_t The data type (e.g., float, double) of the tensor elements.
 * @param pmu Input PackedTensorAccessor64 containing `(pt, eta, phi, E or Mass)`. Assumed shape: `[n_particles, 4]`.
 * @param pmc Output PackedTensorAccessor64 to store the calculated `(px, py, pz, E or Mass)`. Assumed shape: `[n_particles, 4]`.
 */

/**
 * @brief CUDA kernel to transform 4-momentum from (pt, eta, phi, Mass) to (px, py, pz, E).
 *
 * This kernel converts `(pt, eta, phi)` to `(px, py, pz)` and calculates the energy `E`
 * using the relativistic energy-momentum relation: `E = sqrt(px^2 + py^2 + pz^2 + Mass^2)`.
 * It assumes the 4th component of the input `pmu` is Mass.
 * Uses shared memory (`pmx`, `pmt`) for intermediate storage of input components and squared momentum components.
 * Requires the kernel launch configuration to have `blockDim.y` equal to 4.
 * Relies on `px_`, `py_`, `pz_`, `_sqrt`, `_sum` device functions.
 *
 * @tparam scalar_t The data type (e.g., float, double) of the tensor elements.
 * @param pmu Input PackedTensorAccessor64 containing `(pt, eta, phi, Mass)`. Assumed shape: `[n_particles, 4]`.
 * @param pmc Output PackedTensorAccessor64 to store the calculated `(px, py, pz, E)`. Assumed shape: `[n_particles, 4]`.
 */

/**
 * @brief CUDA kernel to calculate the transverse momentum (pt) from Cartesian components.
 *
 * Calculates `pt = sqrt(px^2 + py^2)` for each particle.
 *
 * @tparam scalar_t The data type (e.g., float, double) of the tensor elements.
 * @param px A PackedTensorAccessor64 representing the x-component of momentum (`px`). Assumed shape: `[n_particles, 1]`.
 * @param py A PackedTensorAccessor64 representing the y-component of momentum (`py`). Assumed shape: `[n_particles, 1]`.
 * @param pt A PackedTensorAccessor64 to store the calculated transverse momentum (`pt`). Assumed shape: `[n_particles, 1]`.
 */

/**
 * @brief CUDA kernel to calculate the azimuthal angle (phi) from Cartesian components.
 *
 * Calculates `phi = atan2(py, px)` for each particle.
 *
 * @tparam scalar_t The data type (e.g., float, double) of the tensor elements.
 * @param px A PackedTensorAccessor64 representing the x-component of momentum (`px`). Assumed shape: `[n_particles, 1]`.
 * @param py A PackedTensorAccessor64 representing the y-component of momentum (`py`). Assumed shape: `[n_particles, 1]`.
 * @param phi A PackedTensorAccessor64 to store the calculated azimuthal angle (`phi`). Assumed shape: `[n_particles, 1]`.
 */

/**
 * @brief CUDA kernel to calculate the pseudorapidity (eta) from Cartesian momentum components.
 *
 * Calculates `eta = asinh(pz / pt)`, where `pt = sqrt(px^2 + py^2)`.
 * Assumes the input `pmc` contains `(px, py, pz, ...)` in its first three columns.
 *
 * @tparam scalar_t The data type (e.g., float, double) of the tensor elements.
 * @param pmc A PackedTensorAccessor64 containing Cartesian momentum components `(px, py, pz, ...)`. Assumed shape: `[n_particles, >=3]`.
 * @param eta A PackedTensorAccessor64 to store the calculated pseudorapidity (`eta`). Assumed shape: `[n_particles, 1]`.
 */

/**
 * @brief CUDA kernel to transform 4-momentum from (px, py, pz, E/Mass) to (pt, eta, phi, E/Mass).
 *
 * This kernel converts the first three components of a 4-vector from Cartesian coordinates
 * (px, py, pz) to cylindrical-like coordinates (pt, eta, phi). The fourth component (Energy or Mass)
 * is typically passed through unchanged.
 * It uses shared memory (`pmx`) to cache the input `pmc` components for a thread block.
 * Requires the kernel launch configuration to have `blockDim.y` equal to the number of components (typically 4).
 * Relies on `pt_`, `eta_`, `phi_` device functions for the component calculations. Note that `eta_` calculation internally computes `pt`.
 *
 * @tparam scalar_t The data type (e.g., float, double) of the tensor elements.
 * @param pmc Input PackedTensorAccessor64 containing `(px, py, pz, E or Mass)`. Assumed shape: `[n_particles, 4]`.
 * @param pmu Output PackedTensorAccessor64 to store the calculated `(pt, eta, phi, E or Mass)`. Assumed shape: `[n_particles, 4]`.
 */

/**
 * @brief CUDA kernel to transform 4-momentum from (px, py, pz, Mass) to (pt, eta, phi, E).
 *
 * This kernel converts `(px, py, pz)` to `(pt, eta, phi)` and calculates the energy `E`
 * using the relativistic energy-momentum relation: `E = sqrt(px^2 + py^2 + pz^2 + Mass^2)`.
 * It assumes the 4th component of the input `pmc` is Mass.
 * Uses shared memory (`pmx`, `pmt`) for intermediate storage of input components and squared momentum components.
 * Requires the kernel launch configuration to have `blockDim.y` equal to 4.
 * Relies on `pt_`, `eta_`, `phi_`, `_sqrt`, `_sum` device functions.
 *
 * @tparam scalar_t The data type (e.g., float, double) of the tensor elements.
 * @param pmc Input PackedTensorAccessor64 containing `(px, py, pz, Mass)`. Assumed shape: `[n_particles, 4]`.
 * @param pmu Output PackedTensorAccessor64 to store the calculated `(pt, eta, phi, E)`. Assumed shape: `[n_particles, 4]`.
 */
#include <utils/atomic.cuh> // Assumed to contain device functions like px_, py_, pz_, pt_, eta_, phi_, _sqrt, _sum

// --- Kernel Implementations ---

template <typename scalar_t>
__global__ void PxK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt,
    const torch::PackedTensorAccessor64<scalar_t,
