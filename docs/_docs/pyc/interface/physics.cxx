/**
 * @file physics.cxx
 * @brief This file defines the C++ interface for various physics calculations, primarily focused on particle kinematics, using the libtorch library.
 *
 * It provides functions to compute quantities like momentum squared (P^2), momentum magnitude (P),
 * relativistic beta squared (Beta^2), relativistic beta (Beta), invariant mass squared (M^2),
 * invariant mass (M), transverse mass squared (Mt^2), transverse mass (Mt), polar angle (Theta),
 * and the angular separation DeltaR (ΔR).
 *
 * Functions are organized into namespaces based on the coordinate system (Cartesian or Polar)
 * and the input format ('separate' for individual component tensors or 'combined' for a single
 * tensor containing all components). All calculations operate on torch::Tensor objects,
 * allowing for efficient batch processing and potential GPU acceleration.
 */

#include <torch/torch.h> // Include the libtorch header for tensor operations.

namespace pyc {
/**
 * @namespace pyc
 * @brief Root namespace for the project's C++ components accessible from Python.
 */
namespace physics {
/**
 * @namespace physics
 * @brief Namespace containing physics-related calculations.
 */
namespace cartesian {
/**
 * @namespace cartesian
 * @brief Namespace for calculations using Cartesian coordinates (px, py, pz, e).
 */
namespace separate {
/**
 * @namespace separate
 * @brief Namespace for functions accepting input components as separate tensors.
 */

/**
 * @brief Calculates the square of the momentum magnitude (P^2) from individual Cartesian momentum components.
 * @details This function computes P^2 = px^2 + py^2 + pz^2. It assumes the input tensors represent momentum components in a Cartesian system.
 *          The calculation is performed element-wise if the input tensors have multiple entries.
 * @param px A torch::Tensor representing the x-component(s) of the momentum.
 * @param py A torch::Tensor representing the y-component(s) of the momentum.
 * @param pz A torch::Tensor representing the z-component(s) of the momentum.
 * @return A torch::Tensor containing the calculated square of the momentum magnitude (P^2) for each corresponding input element.
 * @note The input tensors must be broadcastable according to libtorch rules.
 */
torch::Tensor P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

} // namespace separate
namespace combined {
/**
 * @namespace combined
 * @brief Namespace for functions accepting input components combined into a single tensor.
 */

/**
 * @brief Calculates the square of the momentum magnitude (P^2) from a combined Cartesian momentum tensor.
 * @details This function computes P^2 = px^2 + py^2 + pz^2 using components extracted from a single input tensor.
 *          It expects the input tensor `pmc` to have shape (..., 3), where the last dimension contains px, py, and pz respectively.
 * @param pmc A torch::Tensor where the last dimension holds the Cartesian momentum components (px, py, pz).
 *            For example, shape could be (N, 3) for N particles.
 * @return A torch::Tensor containing the calculated square of the momentum magnitude (P^2). The shape will be the input shape minus the last dimension (e.g., (N,) if input was (N, 3)).
 * @note Assumes the last dimension of size 3 corresponds to (px, py, pz).
 */
torch::Tensor P2(torch::Tensor pmc);

} // namespace combined
} // namespace cartesian
namespace polar {
/**
 * @namespace polar
 * @brief Namespace for calculations using Polar/Cylindrical coordinates (pt, eta, phi, e).
 * @details This typically refers to coordinates used in collider physics: transverse momentum (pt), pseudorapidity (eta), and azimuthal angle (phi).
 */
namespace separate {
/**
 * @namespace separate
 * @brief Namespace for functions accepting input components as separate tensors.
 */

/**
 * @brief Calculates the square of the momentum magnitude (P^2) from individual polar momentum components.
 * @details This function computes P^2 = pt^2 * cosh^2(eta). It uses the relationship P = pt * cosh(eta).
 *          The azimuthal angle phi is not needed for calculating the magnitude squared.
 * @param pt A torch::Tensor representing the transverse momentum component(s).
 * @param eta A torch::Tensor representing the pseudorapidity component(s).
 * @param phi A torch::Tensor representing the azimuthal angle component(s). Although required by the signature for consistency, it's not used in the P^2 calculation itself.
 * @return A torch::Tensor containing the calculated square of the momentum magnitude (P^2).
 * @note The input tensors must be broadcastable. Phi is included for potential future use or consistency but isn't mathematically required here.
 */
torch::Tensor P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);

} // namespace separate
namespace combined {
/**
 * @namespace combined
 * @brief Namespace for functions accepting input components combined into a single tensor.
 */

/**
 * @brief Calculates the square of the momentum magnitude (P^2) from a combined polar momentum tensor.
 * @details This function computes P^2 = pt^2 * cosh^2(eta) using components extracted from a single input tensor.
 *          It expects the input tensor `pmu` to have shape (..., 3), where the last dimension contains pt, eta, and phi respectively.
 * @param pmu A torch::Tensor where the last dimension holds the polar momentum components (pt, eta, phi).
 *            For example, shape could be (N, 3) for N particles.
 * @return A torch::Tensor containing the calculated square of the momentum magnitude (P^2). The shape will be the input shape minus the last dimension (e.g., (N,) if input was (N, 3)).
 * @note Assumes the last dimension of size 3 corresponds to (pt, eta, phi). The phi component is not used in this specific calculation.
 */
torch::Tensor P2(torch::Tensor pmu);

} // namespace combined
} // namespace polar
namespace cartesian {
// Re-opening cartesian namespace for P function
namespace separate {

/**
 * @brief Calculates the magnitude of the momentum (P) from individual Cartesian momentum components.
 * @details This function computes P = sqrt(px^2 + py^2 + pz^2). It is the square root of the P2 calculation.
 * @param px A torch::Tensor representing the x-component(s) of the momentum.
 * @param py A torch::Tensor representing the y-component(s) of the momentum.
 * @param pz A torch::Tensor representing the z-component(s) of the momentum.
 * @return A torch::Tensor containing the calculated momentum magnitude (P).
 * @note The input tensors must be broadcastable.
 */
torch::Tensor P(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the magnitude of the momentum (P) from a combined Cartesian momentum tensor.
 * @details This function computes P = sqrt(px^2 + py^2 + pz^2) using components extracted from a single input tensor.
 *          It expects the input tensor `pmc` to have shape (..., 3), where the last dimension contains px, py, and pz respectively.
 * @param pmc A torch::Tensor where the last dimension holds the Cartesian momentum components (px, py, pz).
 * @return A torch::Tensor containing the calculated momentum magnitude (P). The shape will be the input shape minus the last dimension.
 * @note Assumes the last dimension of size 3 corresponds to (px, py, pz).
 */
torch::Tensor P(torch::Tensor pmc);

} // namespace combined
} // namespace cartesian
namespace polar {
// Re-opening polar namespace for P function
namespace separate {

/**
 * @brief Calculates the magnitude of the momentum (P) from individual polar momentum components.
 * @details This function computes P = pt * cosh(eta). It is the square root of the P2 calculation in polar coordinates.
 *          The azimuthal angle phi is not needed.
 * @param pt A torch::Tensor representing the transverse momentum component(s).
 * @param eta A torch::Tensor representing the pseudorapidity component(s).
 * @param phi A torch::Tensor representing the azimuthal angle component(s). (Not used in calculation).
 * @return A torch::Tensor containing the calculated momentum magnitude (P).
 * @note The input tensors must be broadcastable. Phi is included for consistency.
 */
torch::Tensor P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the magnitude of the momentum (P) from a combined polar momentum tensor.
 * @details This function computes P = pt * cosh(eta) using components extracted from a single input tensor.
 *          It expects the input tensor `pmu` to have shape (..., 3), where the last dimension contains pt, eta, and phi respectively.
 * @param pmu A torch::Tensor where the last dimension holds the polar momentum components (pt, eta, phi).
 * @return A torch::Tensor containing the calculated momentum magnitude (P). The shape will be the input shape minus the last dimension.
 * @note Assumes the last dimension of size 3 corresponds to (pt, eta, phi). The phi component is not used.
 */
torch::Tensor P(torch::Tensor pmu);

} // namespace combined
} // namespace polar
namespace cartesian {
// Re-opening cartesian namespace for Beta2 function
namespace separate {

/**
 * @brief Calculates the square of the relativistic beta (β^2) from Cartesian momentum components and energy.
 * @details This function computes β^2 = P^2 / E^2 = (px^2 + py^2 + pz^2) / e^2. Beta represents the velocity of the particle as a fraction of the speed of light (c=1).
 * @param px A torch::Tensor representing the x-component(s) of the momentum.
 * @param py A torch::Tensor representing the y-component(s) of the momentum.
 * @param pz A torch::Tensor representing the z-component(s) of the momentum.
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated square of the relativistic beta (β^2).
 * @note The input tensors must be broadcastable. Requires energy `e` in addition to momentum components. Assumes units where c=1.
 */
torch::Tensor Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the square of the relativistic beta (β^2) from a combined Cartesian four-momentum tensor.
 * @details This function computes β^2 = P^2 / E^2 = (px^2 + py^2 + pz^2) / e^2 using components extracted from a single input tensor.
 *          It expects the input tensor `pmc` to have shape (..., 4), where the last dimension contains px, py, pz, and e respectively.
 * @param pmc A torch::Tensor where the last dimension holds the Cartesian four-momentum components (px, py, pz, e).
 *            For example, shape could be (N, 4) for N particles.
 * @return A torch::Tensor containing the calculated square of the relativistic beta (β^2). The shape will be the input shape minus the last dimension (e.g., (N,) if input was (N, 4)).
 * @note Assumes the last dimension of size 4 corresponds to (px, py, pz, e). Assumes units where c=1.
 */
torch::Tensor Beta2(torch::Tensor pmc);

} // namespace combined
} // namespace cartesian
namespace polar {
// Re-opening polar namespace for Beta2 function
namespace separate {

/**
 * @brief Calculates the square of the relativistic beta (β^2) from polar momentum components and energy.
 * @details This function computes β^2 = P^2 / E^2 = (pt^2 * cosh^2(eta)) / e^2.
 * @param pt A torch::Tensor representing the transverse momentum component(s).
 * @param eta A torch::Tensor representing the pseudorapidity component(s).
 * @param phi A torch::Tensor representing the azimuthal angle component(s). (Not used in calculation).
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated square of the relativistic beta (β^2).
 * @note The input tensors must be broadcastable. Requires energy `e`. Phi is included for consistency. Assumes units where c=1.
 */
torch::Tensor Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the square of the relativistic beta (β^2) from a combined polar four-momentum tensor.
 * @details This function computes β^2 = P^2 / E^2 = (pt^2 * cosh^2(eta)) / e^2 using components extracted from a single input tensor.
 *          It expects the input tensor `pmu` to have shape (..., 4), where the last dimension contains pt, eta, phi, and e respectively.
 * @param pmu A torch::Tensor where the last dimension holds the polar four-momentum components (pt, eta, phi, e).
 *            For example, shape could be (N, 4) for N particles.
 * @return A torch::Tensor containing the calculated square of the relativistic beta (β^2). The shape will be the input shape minus the last dimension (e.g., (N,) if input was (N, 4)).
 * @note Assumes the last dimension of size 4 corresponds to (pt, eta, phi, e). The phi component is not used. Assumes units where c=1.
 */
torch::Tensor Beta2(torch::Tensor pmu);

} // namespace combined
} // namespace polar
namespace cartesian {
// Re-opening cartesian namespace for Beta function
namespace separate {

/**
 * @brief Calculates the relativistic beta (β) from Cartesian momentum components and energy.
 * @details This function computes β = P / E = sqrt(px^2 + py^2 + pz^2) / e. It is the square root of the Beta2 calculation.
 * @param px A torch::Tensor representing the x-component(s) of the momentum.
 * @param py A torch::Tensor representing the y-component(s) of the momentum.
 * @param pz A torch::Tensor representing the z-component(s) of the momentum.
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated relativistic beta (β).
 * @note The input tensors must be broadcastable. Assumes units where c=1.
 */
torch::Tensor Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the relativistic beta (β) from a combined Cartesian four-momentum tensor.
 * @details This function computes β = P / E = sqrt(px^2 + py^2 + pz^2) / e using components extracted from a single input tensor.
 *          It expects the input tensor `pmc` to have shape (..., 4), where the last dimension contains px, py, pz, and e respectively.
 * @param pmc A torch::Tensor where the last dimension holds the Cartesian four-momentum components (px, py, pz, e).
 * @return A torch::Tensor containing the calculated relativistic beta (β). The shape will be the input shape minus the last dimension.
 * @note Assumes the last dimension of size 4 corresponds to (px, py, pz, e). Assumes units where c=1.
 */
torch::Tensor Beta(torch::Tensor pmc);

} // namespace combined
} // namespace cartesian
namespace polar {
// Re-opening polar namespace for Beta function
namespace separate {

/**
 * @brief Calculates the relativistic beta (β) from polar momentum components and energy.
 * @details This function computes β = P / E = (pt * cosh(eta)) / e. It is the square root of the Beta2 calculation in polar coordinates.
 * @param pt A torch::Tensor representing the transverse momentum component(s).
 * @param eta A torch::Tensor representing the pseudorapidity component(s).
 * @param phi A torch::Tensor representing the azimuthal angle component(s). (Not used in calculation).
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated relativistic beta (β).
 * @note The input tensors must be broadcastable. Phi is included for consistency. Assumes units where c=1.
 */
torch::Tensor Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the relativistic beta (β) from a combined polar four-momentum tensor.
 * @details This function computes β = P / E = (pt * cosh(eta)) / e using components extracted from a single input tensor.
 *          It expects the input tensor `pmu` to have shape (..., 4), where the last dimension contains pt, eta, phi, and e respectively.
 * @param pmu A torch::Tensor where the last dimension holds the polar four-momentum components (pt, eta, phi, e).
 * @return A torch::Tensor containing the calculated relativistic beta (β). The shape will be the input shape minus the last dimension.
 * @note Assumes the last dimension of size 4 corresponds to (pt, eta, phi, e). The phi component is not used. Assumes units where c=1.
 */
torch::Tensor Beta(torch::Tensor pmu);

} // namespace combined
} // namespace polar
namespace cartesian {
// Re-opening cartesian namespace for M2 function
namespace separate {

/**
 * @brief Calculates the square of the invariant mass (M^2) from Cartesian momentum components and energy.
 * @details This function computes M^2 = E^2 - P^2 = e^2 - (px^2 + py^2 + pz^2). This is a Lorentz invariant quantity.
 * @param px A torch::Tensor representing the x-component(s) of the momentum.
 * @param py A torch::Tensor representing the y-component(s) of the momentum.
 * @param pz A torch::Tensor representing the z-component(s) of the momentum.
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated square of the invariant mass (M^2).
 * @note The input tensors must be broadcastable. Assumes units where c=1. The result can be negative due to numerical precision or if the input represents a space-like four-vector.
 */
torch::Tensor M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the square of the invariant mass (M^2) from a combined Cartesian four-momentum tensor.
 * @details This function computes M^2 = E^2 - P^2 = e^2 - (px^2 + py^2 + pz^2) using components extracted from a single input tensor.
 *          It expects the input tensor `pmc` to have shape (..., 4), where the last dimension contains px, py, pz, and e respectively.
 * @param pmc A torch::Tensor where the last dimension holds the Cartesian four-momentum components (px, py, pz, e).
 * @return A torch::Tensor containing the calculated square of the invariant mass (M^2). The shape will be the input shape minus the last dimension.
 * @note Assumes the last dimension of size 4 corresponds to (px, py, pz, e). Assumes units where c=1. The result can be negative.
 */
torch::Tensor M2(torch::Tensor pmc);

} // namespace combined
} // namespace cartesian
namespace polar {
// Re-opening polar namespace for M2 function
namespace separate {

/**
 * @brief Calculates the square of the invariant mass (M^2) from polar momentum components and energy.
 * @details This function computes M^2 = E^2 - P^2 = e^2 - (pt^2 * cosh^2(eta)).
 * @param pt A torch::Tensor representing the transverse momentum component(s).
 * @param eta A torch::Tensor representing the pseudorapidity component(s).
 * @param phi A torch::Tensor representing the azimuthal angle component(s). (Not used in calculation).
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated square of the invariant mass (M^2).
 * @note The input tensors must be broadcastable. Phi is included for consistency. Assumes units where c=1. The result can be negative.
 */
torch::Tensor M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the square of the invariant mass (M^2) from a combined polar four-momentum tensor.
 * @details This function computes M^2 = E^2 - P^2 = e^2 - (pt^2 * cosh^2(eta)) using components extracted from a single input tensor.
 *          It expects the input tensor `pmu` to have shape (..., 4), where the last dimension contains pt, eta, phi, and e respectively.
 * @param pmu A torch::Tensor where the last dimension holds the polar four-momentum components (pt, eta, phi, e).
 * @return A torch::Tensor containing the calculated square of the invariant mass (M^2). The shape will be the input shape minus the last dimension.
 * @note Assumes the last dimension of size 4 corresponds to (pt, eta, phi, e). The phi component is not used. Assumes units where c=1. The result can be negative.
 */
torch::Tensor M2(torch::Tensor pmu);

} // namespace combined
} // namespace polar
namespace cartesian {
// Re-opening cartesian namespace for M function
namespace separate {

/**
 * @brief Calculates the invariant mass (M) from Cartesian momentum components and energy.
 * @details This function computes M = sqrt(E^2 - P^2) = sqrt(e^2 - (px^2 + py^2 + pz^2)). It takes the square root of the M2 calculation.
 *          For M^2 < 0, the result might be NaN or handled depending on the sqrt implementation for negative inputs. Often, M is defined as sqrt(abs(M^2)) or requires M^2 >= 0.
 * @param px A torch::Tensor representing the x-component(s) of the momentum.
 * @param py A torch::Tensor representing the y-component(s) of the momentum.
 * @param pz A torch::Tensor representing the z-component(s) of the momentum.
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated invariant mass (M).
 * @note The input tensors must be broadcastable. Assumes units where c=1. Care must be taken for cases where M^2 < 0. Consider using `torch::sqrt(torch::relu(M2(...)))` if negative M^2 should result in M=0.
 */
torch::Tensor M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the invariant mass (M) from a combined Cartesian four-momentum tensor.
 * @details This function computes M = sqrt(E^2 - P^2) = sqrt(e^2 - (px^2 + py^2 + pz^2)) using components extracted from a single input tensor.
 *          It expects the input tensor `pmc` to have shape (..., 4), where the last dimension contains px, py, pz, and e respectively.
 * @param pmc A torch::Tensor where the last dimension holds the Cartesian four-momentum components (px, py, pz, e).
 * @return A torch::Tensor containing the calculated invariant mass (M). The shape will be the input shape minus the last dimension.
 * @note Assumes the last dimension of size 4 corresponds to (px, py, pz, e). Assumes units where c=1. Care must be taken for cases where M^2 < 0.
 */
torch::Tensor M(torch::Tensor pmc);

} // namespace combined
} // namespace cartesian
namespace polar {
// Re-opening polar namespace for M function
namespace separate {

/**
 * @brief Calculates the invariant mass (M) from polar momentum components and energy.
 * @details This function computes M = sqrt(E^2 - P^2) = sqrt(e^2 - (pt^2 * cosh^2(eta))). It takes the square root of the M2 calculation in polar coordinates.
 * @param pt A torch::Tensor representing the transverse momentum component(s).
 * @param eta A torch::Tensor representing the pseudorapidity component(s).
 * @param phi A torch::Tensor representing the azimuthal angle component(s). (Not used in calculation).
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated invariant mass (M).
 * @note The input tensors must be broadcastable. Phi is included for consistency. Assumes units where c=1. Care must be taken for cases where M^2 < 0.
 */
torch::Tensor M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the invariant mass (M) from a combined polar four-momentum tensor.
 * @details This function computes M = sqrt(E^2 - P^2) = sqrt(e^2 - (pt^2 * cosh^2(eta))) using components extracted from a single input tensor.
 *          It expects the input tensor `pmu` to have shape (..., 4), where the last dimension contains pt, eta, phi, and e respectively.
 * @param pmu A torch::Tensor where the last dimension holds the polar four-momentum components (pt, eta, phi, e).
 * @return A torch::Tensor containing the calculated invariant mass (M). The shape will be the input shape minus the last dimension.
 * @note Assumes the last dimension of size 4 corresponds to (pt, eta, phi, e). The phi component is not used. Assumes units where c=1. Care must be taken for cases where M^2 < 0.
 */
torch::Tensor M(torch::Tensor pmu);

} // namespace combined
} // namespace polar
namespace cartesian {
// Re-opening cartesian namespace for Mt2 function
namespace separate {

/**
 * @brief Calculates the square of the transverse mass (Mt^2) from the z-momentum and energy.
 * @details This function computes Mt^2 = E^2 - pz^2. The transverse mass is invariant under Lorentz boosts along the z-axis.
 *          It uses only the energy and the longitudinal momentum component (pz).
 * @param pz A torch::Tensor representing the z-component(s) of the momentum.
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated square of the transverse mass (Mt^2).
 * @note The input tensors must be broadcastable. Assumes units where c=1. This definition implicitly uses E and pz, not transverse components directly.
 */
torch::Tensor Mt2(torch::Tensor pz, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the square of the transverse mass (Mt^2) from a combined Cartesian tensor containing pz and energy.
 * @details This function computes Mt^2 = E^2 - pz^2 using components extracted from a single input tensor.
 *          It expects the input tensor `pmc` to have shape (..., N), where the last dimension contains pz and e at specific indices (convention needed, e.g., indices 2 and 3 if input is px, py, pz, e).
 *          *Correction:* The documentation for the combined Cartesian `Mt2` likely intends to use `px`, `py`, and `e` to calculate `Et^2 = e^2 * (pt^2 / P^2)` or similar, or more commonly `Mt^2 = Et^2 - pt^2`. However, the separate version only uses `pz` and `e`. Clarification on the intended formula for the combined version is needed. Assuming it follows the separate version: E^2 - pz^2.
 * @param pmc A torch::Tensor where the last dimension holds relevant components, including pz and e. The exact indices depend on the convention (e.g., index 2 for pz, index 3 for e if the tensor is [px, py, pz, e]).
 * @return A torch::Tensor containing the calculated square of the transverse mass (Mt^2).
 * @note Assumes units where c=1. The interpretation depends heavily on the assumed structure of `pmc` and the intended physics definition. If following the `separate` version, it only needs pz and e.
 */
torch::Tensor Mt2(torch::Tensor pmc);

} // namespace combined
} // namespace cartesian
namespace polar {
// Re-opening polar namespace for Mt2 function
namespace separate {

/**
 * @brief Calculates the square of the transverse mass (Mt^2) from transverse momentum, pseudorapidity, and energy.
 * @details This function computes Mt^2 = E^2 - pz^2. To get pz from polar coordinates, pz = pt * sinh(eta). So, Mt^2 = e^2 - (pt * sinh(eta))^2.
 *          Alternatively, if Mt is defined using transverse energy Et (where Et^2 = M^2 + pt^2), then Mt^2 = M^2 + pt^2. Using M^2 = E^2 - P^2 = E^2 - pt^2*cosh^2(eta), this gives Mt^2 = E^2 - pt^2*cosh^2(eta) + pt^2 = E^2 - pt^2*(cosh^2(eta) - 1) = E^2 - pt^2*sinh^2(eta). Both definitions lead to the same result.
 * @param pt A torch::Tensor representing the transverse momentum component(s).
 * @param eta A torch::Tensor representing the pseudorapidity component(s).
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated square of the transverse mass (Mt^2).
 * @note The input tensors must be broadcastable. Assumes units where c=1. Phi is not needed.
 */
torch::Tensor Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the square of the transverse mass (Mt^2) from a combined polar tensor containing pt, eta, and energy.
 * @details This function computes Mt^2 = e^2 - (pt * sinh(eta))^2 using components extracted from a single input tensor.
 *          It expects the input tensor `pmu` to have shape (..., N), where the last dimension contains pt, eta, and e at specific indices (convention needed, e.g., indices 0, 1, and 3 if input is pt, eta, phi, e).
 * @param pmu A torch::Tensor where the last dimension holds relevant components, including pt, eta, and e. The exact indices depend on the convention (e.g., index 0 for pt, index 1 for eta, index 3 for e if the tensor is [pt, eta, phi, e]).
 * @return A torch::Tensor containing the calculated square of the transverse mass (Mt^2). The shape will be the input shape minus the last dimension.
 * @note Assumes units where c=1. Assumes the last dimension contains pt, eta, and e according to some convention. Phi component is not used.
 */
torch::Tensor Mt2(torch::Tensor pmu);

} // namespace combined
} // namespace polar
namespace cartesian {
// Re-opening cartesian namespace for Mt function
namespace separate {

/**
 * @brief Calculates the transverse mass (Mt) from the z-momentum and energy.
 * @details This function computes Mt = sqrt(E^2 - pz^2). It is the square root of the Mt2 calculation.
 * @param pz A torch::Tensor representing the z-component(s) of the momentum.
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated transverse mass (Mt).
 * @note The input tensors must be broadcastable. Assumes units where c=1. Care must be taken for cases where Mt^2 < 0.
 */
torch::Tensor Mt(torch::Tensor pz, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the transverse mass (Mt) from a combined Cartesian tensor containing pz and energy.
 * @details This function computes Mt = sqrt(E^2 - pz^2) using components extracted from a single input tensor.
 *          It expects the input tensor `pmc` to have shape (..., N), requiring pz and e components.
 *          *See note in combined Mt2 about ambiguity.* Assuming it follows the separate version: sqrt(E^2 - pz^2).
 * @param pmc A torch::Tensor where the last dimension holds relevant components, including pz and e according to some convention.
 * @return A torch::Tensor containing the calculated transverse mass (Mt).
 * @note Assumes units where c=1. Interpretation depends on the assumed structure of `pmc`. Care must be taken for cases where Mt^2 < 0.
 */
torch::Tensor Mt(torch::Tensor pmc);

} // namespace combined
} // namespace cartesian
namespace polar {
// Re-opening polar namespace for Mt function
namespace separate {

/**
 * @brief Calculates the transverse mass (Mt) from pt, eta, and energy.
 * @details This function computes Mt = sqrt(e^2 - (pt * sinh(eta))^2). It is the square root of the Mt2 calculation in polar coordinates.
 * @param pt A torch::Tensor representing the transverse momentum component(s).
 * @param eta A torch::Tensor representing the pseudorapidity component(s).
 * @param e A torch::Tensor representing the total energy component(s).
 * @return A torch::Tensor containing the calculated transverse mass (Mt).
 * @note The input tensors must be broadcastable. Assumes units where c=1. Phi is not needed. Care must be taken for cases where Mt^2 < 0.
 */
torch::Tensor Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the transverse mass (Mt) from a combined polar tensor containing pt, eta, and energy.
 * @details This function computes Mt = sqrt(e^2 - (pt * sinh(eta))^2) using components extracted from a single input tensor.
 *          It expects the input tensor `pmu` to have shape (..., N), requiring pt, eta, and e components according to some convention.
 * @param pmu A torch::Tensor where the last dimension holds relevant components, including pt, eta, and e according to some convention.
 * @return A torch::Tensor containing the calculated transverse mass (Mt). The shape will be the input shape minus the last dimension.
 * @note Assumes units where c=1. Assumes the last dimension contains pt, eta, and e. Phi component is not used. Care must be taken for cases where Mt^2 < 0.
 */
torch::Tensor Mt(torch::Tensor pmu);

} // namespace combined
} // namespace polar
namespace cartesian {
// Re-opening cartesian namespace for Theta function
namespace separate {

/**
 * @brief Calculates the polar angle (theta, θ) from Cartesian momentum components.
 * @details This function computes the angle with respect to the positive z-axis. θ = atan2(sqrt(px^2 + py^2), pz). The range is typically [0, π].
 *          Note that sqrt(px^2 + py^2) is the transverse momentum, pt. So θ = atan2(pt, pz).
 * @param px A torch::Tensor representing the x-component(s) of the momentum.
 * @param py A torch::Tensor representing the y-component(s) of the momentum.
 * @param pz A torch::Tensor representing the z-component(s) of the momentum.
 * @return A torch::Tensor containing the calculated polar angle (θ) in radians.
 * @note The input tensors must be broadcastable. Uses `atan2` for quadrant correctness.
 */
torch::Tensor Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the polar angle (theta, θ) from a combined Cartesian momentum tensor.
 * @details This function computes θ = atan2(sqrt(px^2 + py^2), pz) using components extracted from a single input tensor.
 *          It expects the input tensor `pmc` to have shape (..., 3), where the last dimension contains px, py, and pz respectively.
 * @param pmc A torch::Tensor where the last dimension holds the Cartesian momentum components (px, py, pz).
 * @return A torch::Tensor containing the calculated polar angle (θ) in radians. The shape will be the input shape minus the last dimension.
 * @note Assumes the last dimension of size 3 corresponds to (px, py, pz). Uses `atan2`.
 */
torch::Tensor Theta(torch::Tensor pmc);

} // namespace combined
} // namespace cartesian
namespace polar {
// Re-opening polar namespace for Theta function
namespace separate {

/**
 * @brief Calculates the polar angle (theta, θ) from polar momentum components.
 * @details This function computes the angle with respect to the positive z-axis using the relationship between pseudorapidity (eta) and theta: eta = -ln(tan(θ/2)).
 *          Solving for theta gives θ = 2 * atan(exp(-eta)).
 * @param pt A torch::Tensor representing the transverse momentum component(s). (Not used in calculation).
 * @param eta A torch::Tensor representing the pseudorapidity component(s).
 * @param phi A torch::Tensor representing the azimuthal angle component(s). (Not used in calculation).
 * @return A torch::Tensor containing the calculated polar angle (θ) in radians, in the range [0, π].
 * @note The input tensors must be broadcastable. Only eta is required for this calculation. pt and phi are included for consistency.
 */
torch::Tensor Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the polar angle (theta, θ) from a combined polar momentum tensor.
 * @details This function computes θ = 2 * atan(exp(-eta)) using the eta component extracted from a single input tensor.
 *          It expects the input tensor `pmu` to have shape (..., 3), where the last dimension contains pt, eta, and phi respectively (eta at index 1).
 * @param pmu A torch::Tensor where the last dimension holds the polar momentum components (pt, eta, phi).
 * @return A torch::Tensor containing the calculated polar angle (θ) in radians. The shape will be the input shape minus the last dimension.
 * @note Assumes the last dimension of size 3 corresponds to (pt, eta, phi), with eta at index 1. pt and phi components are not used.
 */
torch::Tensor Theta(torch::Tensor pmu);

} // namespace combined
} // namespace polar
namespace cartesian {
// Re-opening cartesian namespace for DeltaR function
namespace separate {

/**
 * @brief Calculates the angular separation (Delta R, ΔR) between two particles using Cartesian momentum components.
 * @details This function first calculates the pseudorapidity (eta) and azimuthal angle (phi) for each particle from their Cartesian components,
 *          and then computes ΔR = sqrt( (eta1 - eta2)^2 + (Δphi)^2 ), where Δphi is the smallest difference between phi1 and phi2, handled correctly across the -π to π boundary.
 *          eta = asinh(pz / pt) = asinh(pz / sqrt(px^2 + py^2))
 *          phi = atan2(py, px)
 * @param px1 A torch::Tensor for the x-momentum of the first set of particles.
 * @param px2 A torch::Tensor for the x-momentum of the second set of particles.
 * @param py1 A torch::Tensor for the y-momentum of the first set of particles.
 * @param py2 A torch::Tensor for the y-momentum of the second set of particles.
 * @param pz1 A torch::Tensor for the z-momentum of the first set of particles.
 * @param pz2 A torch::Tensor for the z-momentum of the second set of particles.
 * @return A torch::Tensor containing the calculated ΔR between corresponding particles in the input sets.
 * @note The input tensors must be broadcastable. The calculation involves intermediate steps to find eta and phi for both particles. Handles the phi wrap-around (Δphi = |phi1 - phi2| or 2π - |phi1 - phi2|).
 */
torch::Tensor DeltaR(
    torch::Tensor px1, torch::Tensor px2,
    torch::Tensor py1, torch::Tensor py2,
    torch::Tensor pz1, torch::Tensor pz2
);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the angular separation (Delta R, ΔR) between two particles using combined Cartesian momentum tensors.
 * @details This function computes ΔR = sqrt( (eta1 - eta2)^2 + (Δphi)^2 ) by first extracting px, py, pz for each particle from the combined tensors `pmc1` and `pmc2`,
 *          then calculating their respective eta and phi values, and finally computing ΔR.
 *          It expects `pmc1` and `pmc2` to have shape (..., 3), containing (px, py, pz).
 * @param pmc1 A torch::Tensor for the first set of particles, where the last dimension holds (px, py, pz).
 * @param pmc2 A torch::Tensor for the second set of particles, where the last dimension holds (px, py, pz).
 * @return A torch::Tensor containing the calculated ΔR between corresponding particles. The shape will be the broadcasted input shape minus the last dimension.
 * @note Assumes the last dimension of size 3 corresponds to (px, py, pz). Handles the phi wrap-around.
 */
torch::Tensor DeltaR(torch::Tensor pmc1, torch::Tensor pmc2);

} // namespace combined
} // namespace cartesian
namespace polar {
// Re-opening polar namespace for DeltaR function
namespace separate {

/**
 * @brief Calculates the angular separation (Delta R, ΔR) between two particles using polar coordinates (eta, phi).
 * @details This function computes ΔR = sqrt( (eta1 - eta2)^2 + (Δphi)^2 ), where Δphi is the smallest difference between phi1 and phi2, handled correctly across the -π to π boundary.
 *          This is the standard definition of ΔR in collider physics.
 * @param eta1 A torch::Tensor for the pseudorapidity of the first set of particles.
 * @param eta2 A torch::Tensor for the pseudorapidity of the second set of particles.
 * @param phi1 A torch::Tensor for the azimuthal angle (in radians) of the first set of particles.
 * @param phi2 A torch::Tensor for the azimuthal angle (in radians) of the second set of particles.
 * @return A torch::Tensor containing the calculated ΔR between corresponding particles.
 * @note The input tensors must be broadcastable. Handles the phi wrap-around. pt is not needed for ΔR.
 */
torch::Tensor DeltaR(
    torch::Tensor eta1, torch::Tensor eta2,
    torch::Tensor phi1, torch::Tensor phi2
);

} // namespace separate
namespace combined {

/**
 * @brief Calculates the angular separation (Delta R, ΔR) between two particles using combined polar momentum tensors.
 * @details This function computes ΔR = sqrt( (eta1 - eta2)^2 + (Δphi)^2 ) by extracting eta and phi for each particle from the combined tensors `pmu1` and `pmu2`.
 *          It expects `pmu1` and `pmu2` to have shape (..., N), where the last dimension contains eta and phi at specific indices (e.g., indices 1 and 2 if input is pt, eta, phi).
 * @param pmu1 A torch::Tensor for the first set of particles, where the last dimension holds polar components including eta and phi.
 * @param pmu2 A torch::Tensor for the second set of particles, where the last dimension holds polar components including eta and phi.
 * @return A torch::Tensor containing the calculated ΔR between corresponding particles. The shape will be the broadcasted input shape minus the last dimension.
 * @note Assumes the last dimension contains eta and phi according to some convention (e.g., indices 1 and 2). Handles the phi wrap-around. pt component is not used.
 */
torch::Tensor DeltaR(torch::Tensor pmu1, torch::Tensor pmu2);

} // namespace combined
} // namespace polar
} // namespace physics
} // namespace pyc
