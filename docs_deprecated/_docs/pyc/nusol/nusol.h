#ifndef NUSOL_H
#define NUSOL_H

#include <map>
#include <tuple>
#include <string>
#include <torch/torch.h>

/**
 * @namespace nusol_
 * @brief Provides functions for reconstructing the four-momenta of neutrinos in high-energy physics events, primarily focusing on decays involving W bosons (e.g., top quark decays).
 *
 * @details This namespace encapsulates the algorithms necessary to solve for the unmeasured neutrino momenta using kinematic constraints, such as the W boson mass, and measurements like lepton momenta, b-jet momenta, and missing transverse energy (MET). The functions typically operate on batches of events represented as torch::Tensor objects.
 */
namespace nusol_ {

        /**
         * @brief Computes the base matrix 'M' used in the analytical neutrino reconstruction algorithm.
         * @details This matrix is derived from the kinematics of the lepton and b-quark system. It forms part of the quadratic equation system that needs to be solved to find the neutrino momentum components. The calculation involves components of the lepton and b-quark four-momenta and the assumed masses (W boson, top quark).
         * @param pmc_b A pointer to a torch::Tensor representing the four-momenta (E, px, py, pz) of the b-quarks for a batch of events. Expected shape: [batch_size, 4].
         * @param pmc_mu A pointer to a torch::Tensor representing the four-momenta (E, px, py, pz) of the charged leptons (muons or electrons) for the same batch of events. Expected shape: [batch_size, 4].
         * @param masses A pointer to a torch::Tensor containing relevant particle masses, typically [W mass, top mass]. These are used to enforce kinematic constraints. Expected shape: [2] or potentially [batch_size, 2] if masses vary per event.
         * @return A torch::Tensor representing the computed base matrix 'M' for each event in the batch. The exact dimensions depend on the specific formulation of the algorithm but it encapsulates kinematic information. Expected shape might be [batch_size, 3, 3] or similar, depending on the internal representation.
         * @note The input tensors pointed to should contain valid four-momentum data. The function assumes a specific convention (e.g., E, px, py, pz).
         */
        torch::Tensor BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses);

        /**
         * @brief Calculates the parameters of the hyperbola representing the possible transverse momentum solutions for the neutrino.
         * @details Based on the W boson mass constraint (m_W^2 = (p_lepton + p_neutrino)^2) and the measured lepton momentum, the possible solutions for the neutrino's transverse momentum (nu_px, nu_py) lie on a hyperbola (or an ellipse, depending on the formulation). This function computes the parameters defining this hyperbola for each event.
         * @param pmc_b A pointer to a torch::Tensor representing the four-momenta (E, px, py, pz) of the b-quarks. While not directly used for the W constraint itself, it might be needed if the calculation is embedded within a top-quark decay context or for consistency with other functions. Expected shape: [batch_size, 4].
         * @param pmc_mu A pointer to a torch::Tensor representing the four-momenta (E, px, py, pz) of the charged leptons. This is crucial for defining the hyperbola. Expected shape: [batch_size, 4].
         * @param masses A pointer to a torch::Tensor containing relevant particle masses, primarily the W boson mass. Expected shape: [>=1], where the first element is typically m_W.
         * @return A torch::Tensor containing the parameters that define the hyperbola (e.g., coefficients of the quadratic form in nu_px, nu_py) for each event. Expected shape might be [batch_size, num_params].
         * @see Intersection
         */
        torch::Tensor Hperp(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses);

        /**
         * @brief Computes the intersection points of two geometric objects, typically conic sections (ellipses/hyperbolas), representing kinematic constraints.
         * @details In neutrino reconstruction, constraints from the W mass (often forming a hyperbola/ellipse in the nu_px, nu_py plane) and the MET measurement (often modeled as an ellipse representing the probability distribution of the sum of neutrino transverse momenta) need to be combined. This function finds the intersection points of the geometric shapes defined by tensors A and B.
         * @param A A pointer to a torch::Tensor representing the parameters of the first geometric object (e.g., the hyperbola from `Hperp` or a MET ellipse). Expected shape depends on the representation, e.g., [batch_size, num_params_A].
         * @param B A pointer to a torch::Tensor representing the parameters of the second geometric object (e.g., another hyperbola/ellipse, potentially related to MET or another decay chain). Expected shape depends on the representation, e.g., [batch_size, num_params_B].
         * @param nulls A double value used as a tolerance or threshold for numerical stability, particularly when dealing with near-tangent cases or matrix inversions/determinants close to zero. It helps handle degeneracies or ill-conditioned problems.
         * @return A std::tuple containing two torch::Tensor objects. These typically represent the coordinates (e.g., px, py) of the intersection points. There can be 0, 1, 2, 3, or 4 real intersection points depending on the conic sections. The tensors might store these solutions, potentially padded or flagged if fewer solutions exist. For example, `std::get<0>(result)` could be the x-coordinates and `std::get<1>(result)` the y-coordinates, possibly with shapes like [batch_size, max_solutions].
         * @note The interpretation of tensors A and B depends heavily on the specific reconstruction algorithm being implemented.
         */
        std::tuple<torch::Tensor, torch::Tensor> Intersection(torch::Tensor* A, torch::Tensor* B, double nulls);

        /**
         * @brief Performs the full reconstruction of a single neutrino's four-momentum.
         * @details This function combines the kinematic constraints (W mass) with measured inputs (lepton, b-jet, MET) to solve for the neutrino's four-momentum (E, px, py, pz). It typically involves solving a quadratic equation for the neutrino's pz component, potentially leading to zero, one (complex), or two real solutions per event. MET information is used to constrain the transverse components (px, py).
         * @param pmc_b A pointer to a torch::Tensor representing the b-quark four-momenta. Expected shape: [batch_size, 4].
         * @param pmc_mu A pointer to a torch::Tensor representing the lepton four-momenta. Expected shape: [batch_size, 4].
         * @param met_xy A pointer to a torch::Tensor representing the measured missing transverse energy vector (MET_x, MET_y). Expected shape: [batch_size, 2].
         * @param masses A pointer to a torch::Tensor containing relevant particle masses, typically [W mass, top mass]. Expected shape: [2] or potentially [batch_size, 2].
         * @param sigma A pointer to a torch::Tensor representing the covariance matrix or uncertainties associated with the MET measurement. This is used to incorporate MET resolution effects, often by defining an elliptical constraint in the (nu_px, nu_py) plane or by calculating a chi-squared value. Expected shape: [batch_size, 2, 2] or similar representation of uncertainty.
         * @param null A double value used as a tolerance for numerical calculations, similar to the `nulls` parameter in `Intersection`, potentially for checking if the discriminant of the quadratic equation is close to zero.
         * @return A std::map where keys are strings describing the output quantities and values are torch::Tensor objects. Common entries include:
         *         - "Nu": Tensor containing the reconstructed neutrino four-momenta solutions. Shape might be [batch_size, num_solutions, 4], where num_solutions is often 2.
         *         - "discriminant": Tensor indicating the nature of the solutions (e.g., positive for two real solutions, zero for one, negative for complex). Shape: [batch_size].
         *         - "chi2": Tensor containing chi-squared values if MET uncertainties are used to evaluate the goodness-of-fit of the solutions. Shape: [batch_size, num_solutions].
         *         - Other intermediate or diagnostic quantities.
         * @note The function might return complex momentum solutions if the kinematics are inconsistent within measurement uncertainties. The handling of such cases (e.g., taking the real part, discarding) depends on the downstream analysis.
         */
        std::map<std::string, torch::Tensor> Nu(
                                        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy,
                                        torch::Tensor* masses, torch::Tensor* sigma , double null);

        /**
         * @brief Performs the reconstruction of two neutrino four-momenta simultaneously, typically for dileptonic ttbar events.
         * @details In events with two W bosons decaying leptonically (e.g., ttbar -> (b l nu) (b l nu)), there are two neutrinos contributing to the total MET. This function attempts to solve the system of equations arising from two W mass constraints and the single MET measurement to find the four-momenta of both neutrinos. This is generally an underconstrained problem, often requiring assumptions, scanning, or approximations.
         * @param pmc_b1 A pointer to a torch::Tensor for the first b-quark four-momenta. Expected shape: [batch_size, 4].
         * @param pmc_b2 A pointer to a torch::Tensor for the second b-quark four-momenta. Expected shape: [batch_size, 4].
         * @param pmc_mu1 A pointer to a torch::Tensor for the first lepton four-momenta. Expected shape: [batch_size, 4].
         * @param pmc_mu2 A pointer to a torch::Tensor for the second lepton four-momenta. Expected shape: [batch_size, 4].
         * @param met_xy A pointer to a torch::Tensor for the total measured MET vector (MET_x, MET_y). Expected shape: [batch_size, 2].
         * @param null A double value used as a tolerance for numerical calculations, similar to the `null` parameter in `Nu`.
         * @param m1 A pointer to a torch::Tensor containing the mass constraint(s) for the first decay chain (e.g., [W mass, top mass]). Expected shape: [2] or similar.
         * @param m2 A pointer to a torch::Tensor containing the mass constraint(s) for the second decay chain. If set to `nullptr` (default), it's assumed to be the same as `m1`. Expected shape: [2] or similar.
         * @return A std::map where keys are strings and values are torch::Tensor objects. The exact content depends heavily on the specific algorithm implemented (as NuNu reconstruction has various approaches). Potential entries:
         *         - "Nu1": Tensor with solutions for the first neutrino's four-momentum.
         *         - "Nu2": Tensor with solutions for the second neutrino's four-momentum.
         *         - "chi2": Tensor with chi-squared or likelihood values associated with the solutions.
         *         - "weight": Tensors representing weights if a scanning or sampling method is used.
         *         - Diagnostic flags or parameters related to the solution method.
         *         The shapes will vary based on the algorithm (e.g., number of solutions found per event).
         * @warning Di-neutrino reconstruction is complex and often involves ambiguities or requires assumptions/approximations. The specific implementation details significantly affect the output and its interpretation.
         * @see Nu
         */
        std::map<std::string, torch::Tensor> NuNu(
                                        torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
                                        torch::Tensor* met_xy, double null, torch::Tensor* m1, torch::Tensor* m2 = nullptr);
} // namespace nusol_
#endif // NUSOL_H
