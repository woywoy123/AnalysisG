/**
 * @file transform.cuh
 * @brief Header file for coordinate transformation functions using CUDA and LibTorch.
 *
 * This file declares functions within the `transform_` namespace for converting
 * between different representations of particle kinematics, primarily between
 * Cartesian coordinates (Px, Py, Pz, E) and cylindrical/pseudorapidity coordinates
 * (Pt, eta, phi, E). These functions operate on torch::Tensor objects, implying
 * batch processing capabilities, likely accelerated on a GPU via CUDA.
 */

/**
 * @namespace transform_
 * @brief Contains functions for kinematic variable transformations.
 */

/**
 * @brief Calculates the x-component of momentum (Px).
 * @param pt Pointer to a tensor containing transverse momentum values.
 * @param phi Pointer to a tensor containing azimuthal angle (phi) values.
 * @return A tensor containing the calculated Px values.
 */

/**
 * @brief Calculates the y-component of momentum (Py).
 * @param pt Pointer to a tensor containing transverse momentum values.
 * @param phi Pointer to a tensor containing azimuthal angle (phi) values.
 * @return A tensor containing the calculated Py values.
 */

/**
 * @brief Calculates the z-component of momentum (Pz).
 * @param pt Pointer to a tensor containing transverse momentum values.
 * @param eta Pointer to a tensor containing pseudorapidity (eta) values.
 * @return A tensor containing the calculated Pz values.
 */

/**
 * @brief Calculates Cartesian momentum components (Px, Py, Pz) from Pt, eta, and phi.
 * @param pt Pointer to a tensor containing transverse momentum values.
 * @param eta Pointer to a tensor containing pseudorapidity (eta) values.
 * @param phi Pointer to a tensor containing azimuthal angle (phi) values.
 * @return A tensor containing the calculated [Px, Py, Pz] values, likely stacked along a new dimension.
 */

/**
 * @brief Calculates the 4-momentum in Cartesian coordinates (Px, Py, Pz, E) from Pt, eta, phi, and E.
 * @param pt Pointer to a tensor containing transverse momentum values.
 * @param eta Pointer to a tensor containing pseudorapidity (eta) values.
 * @param phi Pointer to a tensor containing azimuthal angle (phi) values.
 * @param energy Pointer to a tensor containing energy values.
 * @return A tensor containing the calculated [Px, Py, Pz, E] values, likely stacked along a new dimension.
 */

/**
 * @brief Extracts or calculates Cartesian momentum components (Px, Py, Pz) from a 4-momentum tensor.
 * @param pmu Pointer to a tensor containing 4-momentum values (e.g., [Px, Py, Pz, E] or [Pt, eta, phi, E/M]).
 *            The exact expected format depends on the implementation.
 * @return A tensor containing the [Px, Py, Pz] values.
 */

/**
 * @brief Extracts or calculates the 4-momentum in Cartesian coordinates (Px, Py, Pz, E) from a 4-momentum tensor.
 * @param pmu Pointer to a tensor containing 4-momentum values (e.g., [Px, Py, Pz, E] or [Pt, eta, phi, E/M]).
 *            The exact expected format depends on the implementation.
 * @return A tensor containing the [Px, Py, Pz, E] values.
 */

/**
 * @brief Calculates Pt, eta, and phi from Cartesian momentum components stored in a tensor.
 * @param pmc Pointer to a tensor containing Cartesian momentum values (likely [Px, Py, Pz]).
 * @return A tensor containing the calculated [Pt, eta, phi] values, likely stacked along a new dimension.
 */

/**
 * @brief Calculates Pt, eta, and phi from individual Cartesian momentum components.
 * @param px Pointer to a tensor containing Px values.
 * @param py Pointer to a tensor containing Py values.
 * @param pz Pointer to a tensor containing Pz values.
 * @return A tensor containing the calculated [Pt, eta, phi] values, likely stacked along a new dimension.
 */

/**
 * @brief Calculates Pt, eta, phi, and E from a 4-momentum tensor in Cartesian coordinates.
 * @param pmc Pointer to a tensor containing 4-momentum values (likely [Px, Py, Pz, E]).
 * @return A tensor containing the calculated [Pt, eta, phi, E] values, likely stacked along a new dimension.
 */

/**
 * @brief Calculates Pt, eta, phi, and E from individual Cartesian momentum components and energy.
 * @param px Pointer to a tensor containing Px values.
 * @param py Pointer to a tensor containing Py values.
 * @param pz Pointer to a tensor containing Pz values.
 * @param e Pointer to a tensor containing energy (E) values.
 * @return A tensor containing the calculated [Pt, eta, phi, E] values, likely stacked along a new dimension.
 */

/**
 * @brief Calculates the transverse momentum (Pt).
 * @param px Pointer to a tensor containing Px values.
 * @param py Pointer to a tensor containing Py values.
 * @return A tensor containing the calculated Pt values.
 */

/**
 * @brief Calculates the azimuthal angle (phi).
 * @param px Pointer to a tensor containing Px values.
 * @param py Pointer to a tensor containing Py values.
 * @return A tensor containing the calculated phi values.
 */

/**
 * @brief Calculates the pseudorapidity (eta) from Cartesian momentum components stored in a tensor.
 * @param pmc Pointer to a tensor containing Cartesian momentum values (likely [Px, Py, Pz]).
 * @return A tensor containing the calculated eta values.
 */

/**
 * @brief Calculates the pseudorapidity (eta) from individual Cartesian momentum components.
 * @param px Pointer to a tensor containing Px values.
 * @param py Pointer to a tensor containing Py values.
 * @param pz Pointer to a tensor containing Pz values.
 * @return A tensor containing the calculated eta values.
 */
