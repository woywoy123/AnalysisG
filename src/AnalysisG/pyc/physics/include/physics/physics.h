/**
 * @file physics.h
 * @brief High-performance physics calculations for particle kinematics
 * @defgroup pyc_physics Physics Calculations
 * @ingroup module_pyc
 * @{
 *
 * This module provides optimized implementations of common high energy physics
 * calculations, including:
 * - Invariant mass and transverse mass
 * - 3-momentum magnitude and Lorentz factor β
 * - Angular separation (ΔR) between particles
 * - Polar angle θ
 *
 * ## Features
 *
 * - **CUDA Accelerated**: GPU implementations for large-scale processing
 * - **Vectorized**: Batch operations on tensor inputs
 * - **Coordinate Agnostic**: Works with both Cartesian (px,py,pz,E) and combined 4-vectors
 * - **Zero-Copy**: Direct LibTorch tensor operations
 *
 * ## Coordinate Conventions
 *
 * Functions accept two input formats:
 * 1. **Separated**: Individual tensors for px, py, pz, E
 * 2. **Combined**: Single tensor [N x 4] with columns [px, py, pz, E] or [pt, eta, phi, E]
 *
 * @note All calculations use natural units where c = 1
 * @note Input tensors must have compatible shapes for broadcasting
 *
 * ## Usage Example
 *
 * @code{.cpp}
 * #include <physics/physics.h>
 * #include <torch/torch.h>
 *
 * // Separated components
 * torch::Tensor px = torch::tensor({50.0, 30.0});
 * torch::Tensor py = torch::tensor({20.0, 40.0});
 * torch::Tensor pz = torch::tensor({10.0, -15.0});
 * torch::Tensor e = torch::tensor({56.0, 53.0});
 *
 * // Calculate invariant mass
 * torch::Tensor mass = physics_::M(&px, &py, &pz, &e);
 *
 * // Or use combined 4-vector
 * torch::Tensor pmc = torch::stack({px, py, pz, e}, 1);
 * torch::Tensor mass2 = physics_::M(&pmc);
 * @endcode
 *
 * @see pyc_transform for coordinate transformations
 * @see pyc_graph for graph operations
 */

#ifndef PHYSICS_H
#define PHYSICS_H

#include <torch/torch.h>

/**
 * @namespace physics_
 * @brief Namespace containing physics calculation functions
 */
namespace physics_ {

    // ========== 3-Momentum Calculations ==========

    /**
     * @brief Calculate squared 3-momentum magnitude p² = px² + py² + pz²
     *
     * @param px X-component of momentum tensor
     * @param py Y-component of momentum tensor  
     * @param pz Z-component of momentum tensor
     * @return Tensor containing p² values in GeV²
     */
    torch::Tensor P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

    /**
     * @brief Calculate squared 3-momentum from combined 4-vector
     *
     * @param pmc Combined momentum tensor [N x 4] with columns [px, py, pz, E]
     * @return Tensor containing p² values in GeV²
     */
    torch::Tensor P2(torch::Tensor* pmc);

    /**
     * @brief Calculate 3-momentum magnitude p = √(px² + py² + pz²)
     *
     * @param px X-component of momentum
     * @param py Y-component of momentum
     * @param pz Z-component of momentum
     * @return Tensor containing p values in GeV
     */
    torch::Tensor P(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

    /**
     * @brief Calculate 3-momentum magnitude from combined vector
     *
     * @param pmc Combined momentum tensor [N x 4]
     * @return Tensor containing p values in GeV
     */
    torch::Tensor P(torch::Tensor* pmc);

    // ========== Lorentz Factor Calculations ==========

    /**
     * @brief Calculate squared Lorentz velocity β² = p²/E² where p is 3-momentum
     *
     * Used in boost calculations and rapidity computations.
     *
     * @param px X-component of momentum
     * @param py Y-component of momentum
     * @param pz Z-component of momentum
     * @param e Energy
     * @return Tensor containing β² values (dimensionless, range [0,1))
     *
     * @note β² → 1 for highly relativistic particles
     * @note β² = 0 for particles at rest
     */
    torch::Tensor Beta2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

    /**
     * @brief Calculate β² from combined 4-vector
     *
     * @param pmc Combined 4-momentum [N x 4] with [px, py, pz, E]
     * @return Tensor containing β² values
     */
    torch::Tensor Beta2(torch::Tensor* pmc);

    /**
     * @brief Calculate Lorentz velocity β = p/E = √(β²)
     *
     * @param px X-component of momentum
     * @param py Y-component of momentum
     * @param pz Z-component of momentum
     * @param e Energy
     * @return Tensor containing β values (dimensionless, range [0,1))
     */
    torch::Tensor Beta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

    /**
     * @brief Calculate β from combined 4-vector
     *
     * @param pmc Combined 4-momentum [N x 4]
     * @return Tensor containing β values
     */
    torch::Tensor Beta(torch::Tensor* pmc);

    // ========== Invariant Mass Calculations ==========

    /**
     * @brief Calculate squared invariant mass m² = E² - p²
     *
     * Fundamental Lorentz-invariant quantity in special relativity.
     * For composite systems, use 4-momentum sum before calling this function.
     *
     * @param px X-component of momentum
     * @param py Y-component of momentum
     * @param pz Z-component of momentum
     * @param e Energy
     * @return Tensor containing m² values in GeV²
     *
     * @note m² can be negative for space-like 4-vectors (indicates error in physics)
     * @note m² = 0 for massless particles (photons, gluons)
     */
    torch::Tensor M2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

    /**
     * @brief Calculate m² from combined 4-vector
     *
     * @param pmc Combined 4-momentum [N x 4]
     * @return Tensor containing m² values in GeV²
     */
    torch::Tensor M2(torch::Tensor* pmc);

    /**
     * @brief Calculate invariant mass m = √(E² - p²) = √(m²)
     *
     * Standard particle mass calculation. For composite particles (e.g., di-lepton
     * systems, jets), first sum the 4-momenta then apply this function.
     *
     * @param px X-component of momentum
     * @param py Y-component of momentum
     * @param pz Z-component of momentum
     * @param e Energy
     * @return Tensor containing m values in GeV
     *
     * @note Returns NaN for negative m² (space-like 4-vectors)
     * @note For composite systems: M(p₁+p₂) ≠ M(p₁) + M(p₂)
     */
    torch::Tensor M(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

    /**
     * @brief Calculate invariant mass from combined 4-vector
     *
     * @param pmc Combined 4-momentum [N x 4]
     * @return Tensor containing m values in GeV
     */
    torch::Tensor M(torch::Tensor* pmc);

    // ========== Transverse Mass Calculations ==========

    /**
     * @brief Calculate squared transverse mass mₜ² = E² - pz²
     *
     * Transverse mass is invariant under boosts along the z-axis (beam direction).
     * Commonly used in analyses with missing transverse energy.
     *
     * @param pz Z-component of momentum (along beam)
     * @param e Energy
     * @return Tensor containing mₜ² values in GeV²
     *
     * @note mₜ² = pt² + m² for single particles
     */
    torch::Tensor Mt2(torch::Tensor* pz, torch::Tensor* e);

    /**
     * @brief Calculate mₜ² from combined 4-vector
     *
     * @param pmc Combined 4-momentum [N x 4]
     * @return Tensor containing mₜ² values in GeV²
     */
    torch::Tensor Mt2(torch::Tensor* pmc);

    /**
     * @brief Calculate transverse mass mₜ = √(E² - pz²)
     *
     * @param pz Z-component of momentum
     * @param e Energy
     * @return Tensor containing mₜ values in GeV
     */
    torch::Tensor Mt(torch::Tensor* pz, torch::Tensor* e);

    /**
     * @brief Calculate transverse mass from combined 4-vector
     *
     * @param pmc Combined 4-momentum [N x 4]
     * @return Tensor containing mₜ values in GeV
     */
    torch::Tensor Mt(torch::Tensor* pmc);

    // ========== Angular Calculations ==========

    /**
     * @brief Calculate polar angle θ from Cartesian coordinates
     *
     * Polar angle from z-axis (beam direction): θ = arctan(pₜ/pz)
     * where pₜ = √(px² + py²)
     *
     * @param px X-component of momentum
     * @param py Y-component of momentum
     * @param pz Z-component of momentum
     * @return Tensor containing θ values in radians [0, π]
     *
     * @note θ = π/2 for particles perpendicular to beam
     * @note Related to pseudorapidity: η = -ln(tan(θ/2))
     */
    torch::Tensor Theta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

    /**
     * @brief Calculate polar angle from combined vector
     *
     * @param pmc Combined momentum [N x 4] (can be Cartesian or polar)
     * @return Tensor containing θ values in radians
     */
    torch::Tensor Theta(torch::Tensor* pmc);

    /**
     * @brief Calculate angular separation ΔR between two particles in η-φ space
     *
     * Standard ATLAS/CMS metric for particle separation:
     * ΔR = √(Δη² + Δφ²)
     *
     * Used for:
     * - Jet clustering
     * - Isolation criteria
     * - Object matching
     *
     * @param pmu1 First particle 4-momentum [N x 4] with [pt, eta, phi, E]
     * @param pmu2 Second particle 4-momentum [N x 4]
     * @return Tensor containing ΔR values (dimensionless, ≥ 0)
     *
     * @note φ differences are automatically wrapped to [-π, π]
     * @note ΔR < 0.4 is a typical isolation criterion
     */
    torch::Tensor DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2);

    /**
     * @brief Calculate ΔR from separated η and φ components
     *
     * @param eta1 Pseudorapidity of first particle
     * @param eta2 Pseudorapidity of second particle
     * @param phi1 Azimuthal angle of first particle (radians)
     * @param phi2 Azimuthal angle of second particle (radians)
     * @return Tensor containing ΔR values
     *
     * @par Example
     * @code{.cpp}
     * // Check if two particles are isolated (ΔR > 0.4)
     * torch::Tensor dr = physics_::DeltaR(&eta1, &eta2, &phi1, &phi2);
     * torch::Tensor isolated = dr > 0.4;
     * @endcode
     */
    torch::Tensor DeltaR(torch::Tensor* eta1, torch::Tensor* eta2, torch::Tensor* phi1, torch::Tensor* phi2);
}

/** @} */ // end of pyc_physics group

#endif
