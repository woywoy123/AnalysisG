/**
 * @file transform.cxx
 * @brief This file defines the interface for coordinate transformations commonly used in high-energy physics,
 * specifically between Cartesian (px, py, pz) and cylindrical (Pt, Eta, Phi) coordinate systems,
 * potentially including energy (E). Functions are provided for both separate tensor inputs and combined tensor inputs.
 * All operations are performed using torch::Tensor for potential GPU acceleration and integration with PyTorch models.
 */

#include <torch/torch.h>

namespace pyc {
/**
 * @brief Namespace containing physics analysis components callable from Python.
 */
namespace transform {
/**
 * @brief Namespace for coordinate transformation functions.
 */
namespace separate {
/**
 * @brief Namespace for transformation functions where each coordinate component (px, py, pz, pt, eta, phi, e)
 * is provided as a separate torch::Tensor.
 */

/**
 * @brief Calculates the transverse momentum (Pt) from Cartesian momentum components px and py.
 * @details Transverse momentum (Pt) is the component of momentum perpendicular to the beam axis (conventionally the z-axis).
 * It is calculated using the Pythagorean theorem in the transverse plane: Pt = sqrt(px^2 + py^2).
 * This function operates element-wise on the input tensors.
 * @param px A tensor representing the x-component of the momentum for one or more particles.
 * @param py A tensor representing the y-component of the momentum for the same particles. Must have the same shape as px.
 * @return A tensor of the same shape as the inputs, containing the calculated transverse momentum (Pt) for each particle.
 */
torch::Tensor Pt(torch::Tensor px, torch::Tensor py);

/**
 * @brief Calculates the pseudorapidity (Eta) from Cartesian momentum components px, py, and pz.
 * @details Pseudorapidity (Eta) is a spatial coordinate describing the angle of a particle relative to the beam axis.
 * It is defined as Eta = -ln(tan(theta/2)), where theta is the polar angle (the angle with respect to the positive z-axis).
 * Theta can be calculated as atan2(sqrt(px^2 + py^2), pz) = atan2(Pt, pz).
 * This function handles the calculation element-wise. Special care might be needed for particles exactly along the beam axis (Pt=0).
 * @param px A tensor representing the x-component of the momentum.
 * @param py A tensor representing the y-component of the momentum. Must have the same shape as px.
 * @param pz A tensor representing the z-component of the momentum. Must have the same shape as px.
 * @return A tensor of the same shape as the inputs, containing the calculated pseudorapidity (Eta) for each particle.
 */
torch::Tensor Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

/**
 * @brief Calculates the azimuthal angle (Phi) from Cartesian momentum components px and py.
 * @details The azimuthal angle (Phi) is the angle in the transverse (x-y) plane, measured from the positive x-axis.
 * It is calculated using the arctangent function: Phi = atan2(py, px).
 * The result is typically in the range [-pi, pi]. This function operates element-wise.
 * @param px A tensor representing the x-component of the momentum.
 * @param py A tensor representing the y-component of the momentum. Must have the same shape as px.
 * @return A tensor of the same shape as the inputs, containing the calculated azimuthal angle (Phi) in radians for each particle.
 */
torch::Tensor Phi(torch::Tensor px, torch::Tensor py);

/**
 * @brief Calculates Pt, Eta, and Phi simultaneously from Cartesian momentum components px, py, and pz.
 * @details This function combines the calculations of Pt, Eta, and Phi for efficiency or convenience.
 * It takes the three Cartesian momentum components as input and returns a single tensor where these three
 * kinematic variables are stacked along a new dimension.
 * @param px A tensor representing the x-component of the momentum.
 * @param py A tensor representing the y-component of the momentum. Must have the same shape as px.
 * @param pz A tensor representing the z-component of the momentum. Must have the same shape as px.
 * @return A tensor containing the calculated Pt, Eta, and Phi values. If the input tensors have shape `S`,
 * the output tensor will have shape `S + [3]`, where the last dimension holds [Pt, Eta, Phi].
 */
torch::Tensor PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

/**
 * @brief Calculates Pt, Eta, Phi, and includes the original Energy (E) from px, py, pz, and E.
 * @details Similar to PtEtaPhi, this function calculates Pt, Eta, and Phi from the Cartesian momentum components,
 * but it also appends the provided energy tensor `e` to the result. This is useful for creating a complete
 * four-vector representation in (Pt, Eta, Phi, E) coordinates.
 * @param px A tensor representing the x-component of the momentum.
 * @param py A tensor representing the y-component of the momentum. Must have the same shape as px.
 * @param pz A tensor representing the z-component of the momentum. Must have the same shape as px.
 * @param e A tensor representing the energy. Must have the same shape as px.
 * @return A tensor containing the calculated Pt, Eta, Phi, and the input E values. If the input tensors have shape `S`,
 * the output tensor will have shape `S + [4]`, where the last dimension holds [Pt, Eta, Phi, E].
 */
torch::Tensor PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

/**
 * @brief Calculates the x-component of momentum (Px) from transverse momentum (pt) and azimuthal angle (phi).
 * @details This function performs the inverse transformation from cylindrical coordinates back to Cartesian coordinates.
 * Px is calculated as Px = pt * cos(phi). This operates element-wise on the input tensors.
 * @param pt A tensor representing the transverse momentum.
 * @param phi A tensor representing the azimuthal angle in radians. Must have the same shape as pt.
 * @return A tensor of the same shape as the inputs, containing the calculated x-component of momentum (Px).
 */
torch::Tensor Px(torch::Tensor pt, torch::Tensor phi);

/**
 * @brief Calculates the y-component of momentum (Py) from transverse momentum (pt) and azimuthal angle (phi).
 * @details This function performs the inverse transformation from cylindrical coordinates back to Cartesian coordinates.
 * Py is calculated as Py = pt * sin(phi). This operates element-wise on the input tensors.
 * @param pt A tensor representing the transverse momentum.
 * @param phi A tensor representing the azimuthal angle in radians. Must have the same shape as pt.
 * @return A tensor of the same shape as the inputs, containing the calculated y-component of momentum (Py).
 */
torch::Tensor Py(torch::Tensor pt, torch::Tensor phi);

/**
 * @brief Calculates the z-component of momentum (Pz) from transverse momentum (pt) and pseudorapidity (eta).
 * @details This function performs the inverse transformation from cylindrical coordinates back to Cartesian coordinates.
 * Pz is related to Pt and Eta via Pz = pt * sinh(eta). This operates element-wise on the input tensors.
 * Recall that Eta = -ln(tan(theta/2)), and tan(theta) = Pt / Pz.
 * @param pt A tensor representing the transverse momentum.
 * @param eta A tensor representing the pseudorapidity. Must have the same shape as pt.
 * @return A tensor of the same shape as the inputs, containing the calculated z-component of momentum (Pz).
 */
torch::Tensor Pz(torch::Tensor pt, torch::Tensor eta);

/**
 * @brief Calculates Px, Py, and Pz simultaneously from pt, eta, and phi.
 * @details This function combines the inverse calculations of Px, Py, and Pz for efficiency or convenience.
 * It takes the transverse momentum, pseudorapidity, and azimuthal angle as input and returns a single tensor
 * where the resulting Cartesian momentum components are stacked along a new dimension.
 * @param pt A tensor representing the transverse momentum.
 * @param eta A tensor representing the pseudorapidity. Must have the same shape as pt.
 * @param phi A tensor representing the azimuthal angle in radians. Must have the same shape as pt.
 * @return A tensor containing the calculated Px, Py, and Pz values. If the input tensors have shape `S`,
 * the output tensor will have shape `S + [3]`, where the last dimension holds [Px, Py, Pz].
 */
torch::Tensor PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);

/**
 * @brief Calculates Px, Py, Pz, and includes the original Energy (E) from pt, eta, phi, and e.
 * @details Similar to PxPyPz, this function calculates Px, Py, and Pz from the cylindrical coordinates,
 * but it also appends the provided energy tensor `e` to the result. This is useful for creating a complete
 * four-vector representation in (Px, Py, Pz, E) coordinates starting from (Pt, Eta, Phi, E).
 * @param pt A tensor representing the transverse momentum.
 * @param eta A tensor representing the pseudorapidity. Must have the same shape as pt.
 * @param phi A tensor representing the azimuthal angle in radians. Must have the same shape as pt.
 * @param e A tensor representing the energy. Must have the same shape as pt.
 * @return A tensor containing the calculated Px, Py, Pz, and the input E values. If the input tensors have shape `S`,
 * the output tensor will have shape `S + [4]`, where the last dimension holds [Px, Py, Pz, E].
 */
torch::Tensor PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);

} // namespace separate

namespace combined {
/**
 * @brief Namespace for transformation functions where coordinate components are combined into a single tensor.
 * It is assumed that the components are stacked along the last dimension of the input tensor.
 * For example, a Cartesian momentum tensor `pmc` might have shape `[N, 3]` or `[N, 4]`, where the last dimension
 * holds `[px, py, pz]` or `[px, py, pz, E]`. Similarly, `pmu` might hold `[pt, eta, phi]` or `[pt, eta, phi, E]`.
 */

/**
 * @brief Calculates the transverse momentum (Pt) from a combined Cartesian momentum tensor.
 * @details Assumes the input tensor `pmc` has shape `[..., N]` where N >= 2.
 * It extracts `px` (from index 0) and `py` (from index 1) along the last dimension
 * and calculates Pt = sqrt(px^2 + py^2).
 * @param pmc The combined momentum tensor, expected to contain at least px and py as the first two components
 * along its last dimension (e.g., `[..., [px, py, ...]]`).
 * @return A tensor containing the calculated transverse momentum (Pt). The shape will be the same as `pmc` excluding the last dimension.
 */
torch::Tensor Pt(torch::Tensor pmc);

/**
 * @brief Calculates the pseudorapidity (Eta) from a combined Cartesian momentum tensor.
 * @details Assumes the input tensor `pmc` has shape `[..., N]` where N >= 3.
 * It extracts `px` (index 0), `py` (index 1), and `pz` (index 2) along the last dimension
 * and calculates Eta = -ln(tan(theta/2)), where theta = atan2(sqrt(px^2 + py^2), pz).
 * @param pmc The combined momentum tensor, expected to contain px, py, and pz as the first three components
 * along its last dimension (e.g., `[..., [px, py, pz, ...]]`).
 * @return A tensor containing the calculated pseudorapidity (Eta). The shape will be the same as `pmc` excluding the last dimension.
 */
torch::Tensor Eta(torch::Tensor pmc);

/**
 * @brief Calculates the azimuthal angle (Phi) from a combined Cartesian momentum tensor.
 * @details Assumes the input tensor `pmc` has shape `[..., N]` where N >= 2.
 * It extracts `px` (index 0) and `py` (index 1) along the last dimension
 * and calculates Phi = atan2(py, px).
 * @param pmc The combined momentum tensor, expected to contain at least px and py as the first two components
 * along its last dimension (e.g., `[..., [px, py, ...]]`).
 * @return A tensor containing the calculated azimuthal angle (Phi) in radians. The shape will be the same as `pmc` excluding the last dimension.
 */
torch::Tensor Phi(torch::Tensor pmc);

/**
 * @brief Calculates Pt, Eta, and Phi from a combined Cartesian momentum tensor.
 * @details Assumes the input tensor `pmc` has shape `[..., N]` where N >= 3.
 * It extracts `px` (index 0), `py` (index 1), and `pz` (index 2) along the last dimension,
 * calculates Pt, Eta, and Phi, and stacks them into a new tensor along the last dimension.
 * @param pmc The combined momentum tensor, expected to contain px, py, and pz as the first three components
 * along its last dimension (e.g., `[..., [px, py, pz, ...]]`).
 * @return A tensor containing the calculated Pt, Eta, and Phi. The output tensor will have shape `[..., 3]`,
 * where the last dimension holds [Pt, Eta, Phi].
 */
torch::Tensor PtEtaPhi(torch::Tensor pmc);

/**
 * @brief Calculates Pt, Eta, Phi, and retains E from a combined Cartesian momentum-energy tensor.
 * @details Assumes the input tensor `pmc` has shape `[..., 4]`.
 * It extracts `px` (index 0), `py` (index 1), `pz` (index 2), and `E` (index 3) along the last dimension.
 * It calculates Pt, Eta, and Phi using px, py, pz, and then stacks Pt, Eta, Phi, and the original E
 * into a new tensor along the last dimension.
 * @param pmc The combined momentum-energy tensor, expected to have shape `[..., 4]` with the last dimension
 * holding `[px, py, pz, E]`.
 * @return A tensor containing the calculated Pt, Eta, Phi, and the input E. The output tensor will have shape `[..., 4]`,
 * where the last dimension holds [Pt, Eta, Phi, E].
 */
torch::Tensor PtEtaPhiE(torch::Tensor pmc);

/**
 * @brief Calculates the x-component of momentum (Px) from a combined (Pt, Eta, Phi[, E]) tensor.
 * @details Assumes the input tensor `pmu` has shape `[..., N]` where N >= 3.
 * It extracts `pt` (assumed index 0) and `phi` (assumed index 2) along the last dimension
 * and calculates Px = pt * cos(phi). Note the assumed index convention [Pt, Eta, Phi, ...].
 * @param pmu The combined tensor, expected to contain at least pt and phi along its last dimension
 * (e.g., `[..., [pt, eta, phi, ...]]`, assuming pt at index 0, phi at index 2).
 * @return A tensor containing the calculated x-component of momentum (Px). The shape will be the same as `pmu` excluding the last dimension.
 */
torch::Tensor Px(torch::Tensor pmu);

/**
 * @brief Calculates the y-component of momentum (Py) from a combined (Pt, Eta, Phi[, E]) tensor.
 * @details Assumes the input tensor `pmu` has shape `[..., N]` where N >= 3.
 * It extracts `pt` (assumed index 0) and `phi` (assumed index 2) along the last dimension
 * and calculates Py = pt * sin(phi). Note the assumed index convention [Pt, Eta, Phi, ...].
 * @param pmu The combined tensor, expected to contain at least pt and phi along its last dimension
 * (e.g., `[..., [pt, eta, phi, ...]]`, assuming pt at index 0, phi at index 2).
 * @return A tensor containing the calculated y-component of momentum (Py). The shape will be the same as `pmu` excluding the last dimension.
 */
torch::Tensor Py(torch::Tensor pmu);

/**
 * @brief Calculates the z-component of momentum (Pz) from a combined (Pt, Eta, Phi[, E]) tensor.
 * @details Assumes the input tensor `pmu` has shape `[..., N]` where N >= 2 (typically N >= 3).
 * It extracts `pt` (assumed index 0) and `eta` (assumed index 1) along the last dimension
 * and calculates Pz = pt * sinh(eta). Note the assumed index convention [Pt, Eta, Phi, ...].
 * @param pmu The combined tensor, expected to contain at least pt and eta along its last dimension
 * (e.g., `[..., [pt, eta, phi, ...]]`, assuming pt at index 0, eta at index 1).
 * @return A tensor containing the calculated z-component of momentum (Pz). The shape will be the same as `pmu` excluding the last dimension.
 */
torch::Tensor Pz(torch::Tensor pmu);

/**
 * @brief Calculates Px, Py, and Pz from a combined (Pt, Eta, Phi[, E]) tensor.
 * @details Assumes the input tensor `pmu` has shape `[..., N]` where N >= 3.
 * It extracts `pt` (index 0), `eta` (index 1), and `phi` (index 2) along the last dimension,
 * calculates Px, Py, and Pz using the inverse transformation formulas, and stacks them into a new tensor.
 * Note the assumed index convention [Pt, Eta, Phi, ...].
 * @param pmu The combined tensor, expected to contain pt, eta, and phi as the first three components
 * along its last dimension (e.g., `[..., [pt, eta, phi, ...]]`).
 * @return A tensor containing the calculated Px, Py, and Pz. The output tensor will have shape `[..., 3]`,
 * where the last dimension holds [Px, Py, Pz].
 */
torch::Tensor PxPyPz(torch::Tensor pmu);

/**
 * @brief Calculates Px, Py, Pz, and retains E from a combined (Pt, Eta, Phi, E) tensor.
 * @details Assumes the input tensor `pmu` has shape `[..., 4]`.
 * It extracts `pt` (index 0), `eta` (index 1), `phi` (index 2), and `E` (index 3) along the last dimension.
 * It calculates Px, Py, and Pz using pt, eta, phi, and then stacks Px, Py, Pz, and the original E
 * into a new tensor along the last dimension. Note the assumed index convention [Pt, Eta, Phi, E].
 * @param pmu The combined tensor, expected to have shape `[..., 4]` with the last dimension
 * holding `[pt, eta, phi, E]`.
 * @return A tensor containing the calculated Px, Py, Pz, and the input E. The output tensor will have shape `[..., 4]`,
 * where the last dimension holds [Px, Py, Pz, E].
 */
torch::Tensor PxPyPzE(torch::Tensor pmu);

} // namespace combined
} // namespace transform
} // namespace pyc
