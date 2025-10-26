/**
 * @file transform.h
 * @brief Coordinate system transformations for particle kinematics
 * @defgroup pyc_transform Coordinate Transformations
 * @ingroup module_pyc
 * @{
 *
 * This module provides high-performance transformations between Cartesian (px, py, pz, E)
 * and cylindrical polar (pt, η, φ, E) coordinate systems used in collider physics.
 *
 * ## Coordinate Systems
 *
 * ### Cartesian Coordinates (px, py, pz, E)
 * - Standard 3-momentum components + energy
 * - Natural for calculations in arbitrary reference frames
 * - Used in Monte Carlo truth records
 *
 * ### Polar Coordinates (pt, η, φ, E)
 * - **pt**: Transverse momentum = √(px² + py²)
 * - **η** (eta): Pseudorapidity = -ln(tan(θ/2))
 * - **φ** (phi): Azimuthal angle = atan2(py, px)
 * - **E**: Energy (same in both systems)
 *
 * ### Why Polar Coordinates?
 *
 * Collider detectors (ATLAS, CMS, etc.) are cylindrical around the beam axis:
 * - Constant η corresponds to constant polar angle θ
 * - Uniform in η-φ space → uniform detector coverage
 * - ΔR = √(Δη² + Δφ²) is the natural distance metric
 * - Transverse quantities independent of longitudinal boosts
 *
 * ## Features
 *
 * - **CUDA Accelerated**: GPU implementations for batch processing
 * - **Vectorized**: Efficient tensor operations
 * - **Bidirectional**: Convert freely between coordinate systems
 * - **Zero-Copy**: Direct LibTorch integration
 *
 * ## Usage Example
 *
 * @code{.cpp}
 * #include <transform/transform.h>
 * #include <torch/torch.h>
 *
 * // Convert polar to Cartesian
 * torch::Tensor pt = torch::tensor({50.0, 30.0});
 * torch::Tensor eta = torch::tensor({0.5, -1.2});
 * torch::Tensor phi = torch::tensor({1.5, -0.8});
 *
 * torch::Tensor momentum = transform_::PxPyPz(&pt, &eta, &phi);
 * // momentum.shape = [2, 3] with columns [px, py, pz]
 *
 * // Convert back to polar
 * torch::Tensor px = momentum.select(1, 0);
 * torch::Tensor py = momentum.select(1, 1);
 * torch::Tensor pz = momentum.select(1, 2);
 *
 * torch::Tensor pt_back = transform_::Pt(&px, &py);
 * torch::Tensor eta_back = transform_::Eta(&px, &py, &pz);
 * @endcode
 *
 * @see pyc_physics for calculations using these coordinates
 * @see pyc_graph for graph operations
 */

#ifndef TRANSFORM_H
#define TRANSFORM_H
#include <torch/torch.h>

/**
 * @namespace transform_
 * @brief Namespace containing coordinate transformation functions
 */
namespace transform_ {

    // ========== Polar → Cartesian Transformations ==========

    /**
     * @brief Convert (pt, φ) to px-component
     *
     * Formula: px = pt × cos(φ)
     *
     * @param pt Transverse momentum in GeV
     * @param phi Azimuthal angle in radians
     * @return Tensor containing px values in GeV
     */
    torch::Tensor Px(torch::Tensor* pt, torch::Tensor* phi);

    /**
     * @brief Convert (pt, φ) to py-component
     *
     * Formula: py = pt × sin(φ)
     *
     * @param pt Transverse momentum in GeV
     * @param phi Azimuthal angle in radians
     * @return Tensor containing py values in GeV
     */
    torch::Tensor Py(torch::Tensor* pt, torch::Tensor* phi);

    /**
     * @brief Convert (pt, η) to pz-component
     *
     * Formula: pz = pt × sinh(η)
     *
     * @param pt Transverse momentum in GeV
     * @param eta Pseudorapidity (dimensionless)
     * @return Tensor containing pz values in GeV
     *
     * @note η → ±∞ corresponds to pz → ±∞ (along beam)
     * @note η = 0 corresponds to pz = 0 (perpendicular to beam)
     */
    torch::Tensor Pz(torch::Tensor* pt, torch::Tensor* eta);

    /**
     * @brief Convert (pt, η, φ) to full 3-momentum (px, py, pz)
     *
     * Convenience function combining Px, Py, Pz transformations.
     *
     * @param pt Transverse momentum in GeV
     * @param eta Pseudorapidity
     * @param phi Azimuthal angle in radians
     * @return Tensor [N x 3] with columns [px, py, pz] in GeV
     *
     * @par Example
     * @code{.cpp}
     * torch::Tensor pxpypz = transform_::PxPyPz(&pt, &eta, &phi);
     * torch::Tensor px = pxpypz.select(1, 0);
     * torch::Tensor py = pxpypz.select(1, 1);
     * torch::Tensor pz = pxpypz.select(1, 2);
     * @endcode
     */
    torch::Tensor PxPyPz(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi);

    /**
     * @brief Convert (pt, η, φ, E) to full 4-momentum (px, py, pz, E)
     *
     * @param pt Transverse momentum in GeV
     * @param eta Pseudorapidity
     * @param phi Azimuthal angle in radians
     * @param energy Energy in GeV
     * @return Tensor [N x 4] with columns [px, py, pz, E] in GeV
     */
    torch::Tensor PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* energy);

    /**
     * @brief Convert combined polar 4-vector to 3-momentum
     *
     * @param pmu Combined tensor [N x 4] with columns [pt, eta, phi, E]
     * @return Tensor [N x 3] with columns [px, py, pz] in GeV
     */
    torch::Tensor PxPyPz(torch::Tensor* pmu);

    /**
     * @brief Convert combined polar 4-vector to Cartesian 4-vector
     *
     * @param pmu Combined tensor [N x 4] with [pt, eta, phi, E]
     * @return Tensor [N x 4] with [px, py, pz, E] in GeV
     */
    torch::Tensor PxPyPzE(torch::Tensor* pmu);

    // ========== Cartesian → Polar Transformations ==========

    /**
     * @brief Calculate transverse momentum from Cartesian components
     *
     * Formula: pt = √(px² + py²)
     *
     * @param px X-component of momentum in GeV
     * @param py Y-component of momentum in GeV
     * @return Tensor containing pt values in GeV
     *
     * @note Always positive by definition
     */
    torch::Tensor Pt(torch::Tensor* px, torch::Tensor* py);

    /**
     * @brief Calculate pseudorapidity from transverse and longitudinal momentum
     *
     * Formula: η = -ln(tan(θ/2)) where θ = atan2(pt, pz)
     * Equivalent to: η = ½ ln((|p| + pz)/(|p| - pz))
     *
     * @param pt Transverse momentum (used with pz overload)
     * @param pz Longitudinal momentum in GeV
     * @return Tensor containing η values (dimensionless)
     *
     * @note η → +∞ for forward direction (pz → +∞)
     * @note η → -∞ for backward direction (pz → -∞)
     * @note η = 0 at 90° from beam axis
     *
     * @warning Division by zero for |p| = pz (numerical protection applied)
     */
    torch::Tensor PtEta(torch::Tensor* pt, torch::Tensor* pz);

    /**
     * @brief Calculate azimuthal angle from Cartesian components
     *
     * Formula: φ = atan2(py, px)
     *
     * @param px X-component of momentum
     * @param py Y-component of momentum
     * @return Tensor containing φ values in radians, range [-π, π]
     *
     * @note φ = 0 along +x axis
     * @note φ = π/2 along +y axis
     */
    torch::Tensor Phi(torch::Tensor* px, torch::Tensor* py);

    /**
     * @brief Extract φ from combined Cartesian 4-vector
     *
     * @param pmc Combined momentum [N x 4] with [px, py, pz, E]
     * @return Tensor containing φ values in radians
     */
    torch::Tensor Phi(torch::Tensor* pmc);

    /**
     * @brief Calculate pseudorapidity from Cartesian 3-momentum
     *
     * @param px X-component of momentum
     * @param py Y-component of momentum
     * @param pz Z-component of momentum
     * @return Tensor containing η values
     *
     * @par Example - Detector acceptance cuts
     * @code{.cpp}
     * torch::Tensor eta = transform_::Eta(&px, &py, &pz);
     * torch::Tensor central = (eta.abs() < 2.5);  // ATLAS central region
     * torch::Tensor forward = (eta.abs() > 2.5) & (eta.abs() < 4.9);  // Forward
     * @endcode
     */
    torch::Tensor Eta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

    /**
     * @brief Extract η from combined Cartesian 4-vector
     *
     * @param pmc Combined momentum [N x 4] with [px, py, pz, E]
     * @return Tensor containing η values
     */
    torch::Tensor Eta(torch::Tensor* pmc);

    /**
     * @brief Convert Cartesian 3-momentum to polar coordinates (pt, η, φ)
     *
     * @param px X-component of momentum
     * @param py Y-component of momentum
     * @param pz Z-component of momentum
     * @return Tensor [N x 3] with columns [pt, eta, phi]
     */
    torch::Tensor PtEtaPhi(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

    /**
     * @brief Convert Cartesian 4-momentum to polar representation
     *
     * @param px X-component of momentum
     * @param py Y-component of momentum
     * @param pz Z-component of momentum
     * @param e Energy
     * @return Tensor [N x 4] with columns [pt, eta, phi, E]
     */
    torch::Tensor PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

    /**
     * @brief Extract (pt, η, φ) from combined Cartesian 4-vector
     *
     * @param pmc Combined momentum [N x 4] with [px, py, pz, E]
     * @return Tensor [N x 3] with [pt, eta, phi]
     */
    torch::Tensor PtEtaPhi(torch::Tensor* pmc);

    /**
     * @brief Convert combined Cartesian to combined polar 4-vector
     *
     * @param pmc Combined momentum [N x 4] with [px, py, pz, E]
     * @return Tensor [N x 4] with [pt, eta, phi, E]
     *
     * @par Performance Note
     * This is the most efficient way to convert full 4-vectors, as it
     * computes all intermediate values only once.
     */
    torch::Tensor PtEtaPhiE(torch::Tensor* pmc);
}

/** @} */ // end of pyc_transform group

#endif
