#ifndef CU_NUSOL_BASE_H
#define CU_NUSOL_BASE_H

#include <map>
#include <string>
#include <torch/torch.h>
#include <utils/atomic.cuh>

namespace nusol_ {
        /**
         * @brief Performs debug calculations related to the base neutrino solution components.
         *
         * This function is intended for debugging purposes. It likely calculates intermediate
         * quantities used in the `BaseMatrix` function, allowing for inspection of the
         * underlying physics calculations or numerical stability checks. The specific tensors
         * returned in the map provide insights into the steps involved in constructing the
         * base matrix for neutrino momentum reconstruction.
         *
         * @param pmc_b A pointer to a torch::Tensor representing the four-momenta (Px, Py, Pz, E)
         *              of the b-quarks. Expected shape: [N, 4], where N is the number of events.
         *              The tensor should contain floating-point data (e.g., float32 or float64).
         * @param pmc_mu A pointer to a torch::Tensor representing the four-momenta (Px, Py, Pz, E)
         *               of the muons. Expected shape: [N, 4], where N is the number of events.
         *               The tensor should contain floating-point data.
         * @param masses A pointer to a torch::Tensor containing the relevant particle masses.
         *               Typically, this would include the top quark mass (mT), W boson mass (mW),
         *               and potentially the neutrino mass (mN), although mN is often assumed to be zero.
         *               Expected shape: [N, 3] or [N, 2] depending on whether mN is included.
         *               The tensor should contain floating-point data.
         * @return A std::map where keys are strings identifying specific debug quantities
         *         (e.g., "intermediate_vector_X", "scalar_product_Y") and values are
         *         torch::Tensor objects containing the calculated results for each event.
         *         The shapes and contents of these tensors depend on the specific debug
         *         calculations being performed.
         */
        std::map<std::string, torch::Tensor> BaseDebug(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses);

        /**
         * @brief Calculates the base matrix components required for neutrino momentum reconstruction.
         *
         * This function computes fundamental geometric and kinematic quantities derived from the
         * visible decay products (b-quark and muon) and particle masses. These quantities form
         * the basis for solving the kinematic equations to determine the neutrino's momentum.
         * The results are typically used as input for the `Nu` function. The masses are provided
         * event-by-event via a tensor.
         *
         * @param pmc_b A pointer to a torch::Tensor representing the four-momenta (Px, Py, Pz, E)
         *              of the b-quarks. Expected shape: [N, 4]. Data type should be float.
         * @param pmc_mu A pointer to a torch::Tensor representing the four-momenta (Px, Py, Pz, E)
         *               of the muons. Expected shape: [N, 4]. Data type should be float.
         * @param masses A pointer to a torch::Tensor containing the particle masses (mT, mW, mN)
         *               for each event. Expected shape: [N, 3] or [N, 2]. Data type should be float.
         *               The order of masses should be consistent (e.g., top mass, W mass, neutrino mass).
         * @return A std::map where keys are strings identifying components of the base calculation
         *         (e.g., "H", "H_perp", "sigma") and values are torch::Tensor objects containing
         *         the calculated results for each event. These tensors represent vectors and scalars
         *         crucial for the neutrino solution.
         */
        std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses);

        /**
         * @brief Calculates the base matrix components using explicit, fixed mass values.
         *
         * This overload of `BaseMatrix` computes the same fundamental quantities as the version
         * taking a mass tensor, but uses explicitly provided, constant values for the top quark (mT),
         * W boson (mW), and neutrino (mN) masses. This is useful when assuming fixed masses
         * across all events.
         *
         * @param pmc_b A pointer to a torch::Tensor representing the four-momenta (Px, Py, Pz, E)
         *              of the b-quarks. Expected shape: [N, 4]. Data type should be float.
         * @param pmc_mu A pointer to a torch::Tensor representing the four-momenta (Px, Py, Pz, E)
         *               of the muons. Expected shape: [N, 4]. Data type should be float.
         * @param mT The mass of the top quark (double precision floating-point value). Applied uniformly to all events.
         * @param mW The mass of the W boson (double precision floating-point value). Applied uniformly to all events.
         * @param mN The mass of the neutrino (double precision floating-point value). Often set to 0. Applied uniformly to all events.
         * @return A std::map similar to the other `BaseMatrix` overload, containing tensors for
         *         base components like "H", "H_perp", "sigma", calculated using the provided fixed masses.
         */
        std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, double mT, double mW, double mN);

        /**
         * @brief Calculates the intersection points of two geometric or kinematic solution sets.
         *
         * This function likely finds the common solutions or intersection points between two sets
         * of results, possibly representing constraints from different parts of an event or
         * different solution methods. The `nulls` parameter specifies a value used to represent
         * invalid or non-physical results within the input tensors.
         *
         * @param A A pointer to a torch::Tensor representing the first set of solutions or data points.
         *          The exact structure depends on the context (e.g., [N, k] where k is the number of solution parameters).
         * @param B A pointer to a torch::Tensor representing the second set of solutions or data points.
         *          Must be compatible in structure with tensor A.
         * @param nulls A double precision value indicating entries in A and B that should be treated
         *              as invalid or null, and thus ignored during the intersection calculation.
         * @return A std::map containing torch::Tensor objects that represent the intersection.
         *         This could include tensors indicating which elements intersect, the coordinates
         *         of the intersection points, or other relevant metrics. The specific keys and tensor
         *         structures depend on the implementation details.
         */
        std::map<std::string, torch::Tensor> Intersection(torch::Tensor* A, torch::Tensor* B, double nulls);

        /**
         * @brief Calculates neutrino kinematics based on pre-calculated base matrix components and MET.
         *
         * This function takes the results from `BaseMatrix` (H, sigma) and the measured Missing
         * Transverse Energy (MET) to solve for the neutrino's momentum components. It typically
         * involves solving a quadratic equation derived from the W boson mass constraint.
         *
         * @param H A pointer to a torch::Tensor representing the 'H' vector component calculated by `BaseMatrix`.
         *          Expected shape: [N, 3] or similar, representing a 3D vector per event.
         * @param sigma A pointer to a torch::Tensor representing the 'sigma' scalar component calculated by `BaseMatrix`.
         *              Expected shape: [N, 1] or [N].
         * @param met_xy A pointer to a torch::Tensor containing the measured Missing Transverse Energy (MET) vector (Px_miss, Py_miss).
         *               Expected shape: [N, 2]. Data type should be float.
         * @param null A double precision value used to represent or identify non-physical or invalid solutions
         *             (e.g., resulting from a negative discriminant in the quadratic equation).
         * @return A std::map where keys identify neutrino kinematic quantities (e.g., "Nu_Px", "Nu_Py", "Nu_Pz", "discriminant", "Solutions")
         *         and values are torch::Tensor objects. There might be two possible solutions for the neutrino Pz,
         *         often returned or indicated within the map. Invalid solutions are typically marked with the `null` value.
         */
        std::map<std::string, torch::Tensor> Nu(torch::Tensor* H, torch::Tensor* sigma, torch::Tensor* met_xy, double null);

        /**
         * @brief Solves for the kinematics of two neutrinos in events with two semi-leptonic top decays (e.g., ttbar -> (b l nu) (b l nu)).
         *
         * This function tackles the more complex scenario of reconstructing two neutrinos simultaneously.
         * It uses the base matrix components (H, H_perp) calculated separately for each top decay leg
         * and the total MET of the event. The solution often involves iterative numerical methods
         * or scanning techniques to find compatible momenta for both neutrinos that satisfy the
         * individual W mass constraints and sum up to the measured MET.
         *
         * @param H1_ A pointer to a torch::Tensor for the H vector of the first decay leg. Shape: [N, 3].
         * @param H1_perp A pointer to a torch::Tensor for the H_perp vector of the first decay leg. Shape: [N, 3].
         * @param H2_ A pointer to a torch::Tensor for the H vector of the second decay leg. Shape: [N, 3].
         * @param H2_perp A pointer to a torch::Tensor for the H_perp vector of the second decay leg. Shape: [N, 3].
         * @param met_xy A pointer to a torch::Tensor containing the total MET vector (Px_miss, Py_miss). Shape: [N, 2].
         * @param null A double value used to indicate failed solutions or non-physical results (default: 10e-10).
         * @param step The step size used in numerical algorithms (e.g., scanning or gradient descent) if applicable (default: 1e-9).
         * @param tolerance The convergence criterion for iterative numerical methods (default: 1e-6).
         * @param timeout The maximum number of iterations allowed for numerical methods before declaring failure (default: 1000).
         * @return A std::map containing torch::Tensor objects representing the kinematics of the two neutrinos.
         *         Keys might include "Nu1_Px", "Nu1_Py", "Nu1_Pz", "Nu2_Px", "Nu2_Py", "Nu2_Pz", "SolutionStatus", etc.
         *         The map structure will depend heavily on the specific algorithm used (e.g., returning best-fit solutions, multiple solutions, or status flags).
         */
        std::map<std::string, torch::Tensor> NuNu(
                        torch::Tensor* H1_, torch::Tensor* H1_perp, torch::Tensor* H2_, torch::Tensor* H2_perp, torch::Tensor* met_xy,
                        double null = 10e-10, const double step = 1e-9, const double tolerance = 1e-6, const unsigned int timeout = 1000
        );

        /**
         * @brief Calculates neutrino kinematics directly from particle momenta and masses (tensor input).
         *
         * This is a convenience function that combines the steps of `BaseMatrix` and `Nu`.
         * It takes the raw particle four-momenta and mass tensor as input, calculates the
         * necessary intermediate base components internally, and then solves for the neutrino kinematics.
         *
         * @param pmc_b A pointer to a torch::Tensor for the b-quark four-momenta. Shape: [N, 4].
         * @param pmc_mu A pointer to a torch::Tensor for the muon four-momenta. Shape: [N, 4].
         * @param met_xy A pointer to a torch::Tensor for the MET vector. Shape: [N, 2].
         * @param masses A pointer to a torch::Tensor containing particle masses (mT, mW, mN) per event. Shape: [N, 3] or [N, 2].
         * @param sigma A pointer to a torch::Tensor. Its role here might be an input constraint or an output;
         *              clarification needed based on implementation. If it's the sigma from BaseMatrix,
         *              it seems redundant if BaseMatrix is calculated internally. It might represent
         *              an uncertainty or a pre-calculated value used in the solution.
         * @param null A double value used to represent or identify non-physical or invalid solutions.
         * @return A std::map containing the calculated neutrino kinematic tensors, similar to the `Nu(H, sigma, met_xy, null)` function.
         */
        std::map<std::string, torch::Tensor> Nu(
                        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, torch::Tensor* masses, torch::Tensor* sigma , double null
        );

        /**
         * @brief Calculates neutrino kinematics directly from particle momenta using explicit mass values.
         *
         * Similar to the previous `Nu` overload, this function combines `BaseMatrix` and `Nu` steps.
         * However, it uses explicitly provided, fixed mass values (massT, massW) instead of a mass tensor.
         * The neutrino mass is likely assumed to be zero implicitly or handled internally.
         *
         * @param pmc_b A pointer to a torch::Tensor for the b-quark four-momenta. Shape: [N, 4].
         * @param pmc_mu A pointer to a torch::Tensor for the muon four-momenta. Shape: [N, 4].
         * @param met_xy A pointer to a torch::Tensor for the MET vector. Shape: [N, 2].
         * @param sigma A pointer to a torch::Tensor. Similar to the previous `Nu` overload, its exact role requires
         *              implementation context. It might be an input constraint or an output related to the solution.
         * @param null A double value used to represent or identify non-physical or invalid solutions.
         * @param massT The fixed mass of the top quark (double).
         * @param massW The fixed mass of the W boson (double).
         * @return A std::map containing the calculated neutrino kinematic tensors, similar to the `Nu(H, sigma, met_xy, null)` function.
         */
        std::map<std::string, torch::Tensor> Nu(
                        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, torch::Tensor* sigma , double null, double massT, double massW
        );

        /**
         * @brief Solves for di-neutrino kinematics directly from particle momenta and mass tensors.
         *
         * This is a convenience function combining the steps of calculating base components (like `BaseMatrix`)
         * for two decay legs and solving the di-neutrino system (like `NuNu`). It takes raw four-momenta
         * for two b-quarks and two muons, the MET, and tensors containing the masses for each leg.
         * Allows for different masses per event for each decay leg if m1 and m2 are distinct tensors.
         *
         * @param pmc_b1 Pointer to tensor for the first b-quark four-momenta. Shape: [N, 4].
         * @param pmc_b2 Pointer to tensor for the second b-quark four-momenta. Shape: [N, 4].
         * @param pmc_mu1 Pointer to tensor for the first muon four-momenta. Shape: [N, 4].
         * @param pmc_mu2 Pointer to tensor for the second muon four-momenta. Shape: [N, 4].
         * @param met_xy Pointer to tensor for the total MET vector. Shape: [N, 2].
         * @param null A double value used to indicate failed solutions or non-physical results.
         * @param m1 Pointer to tensor containing masses (mT, mW, mN) for the first decay leg. Shape: [N, 3] or [N, 2].
         * @param m2 Pointer to tensor containing masses (mT, mW, mN) for the second decay leg. Shape: [N, 3] or [N, 2].
         *           If nullptr (default), the masses from `m1` are likely used for the second leg as well.
         * @param step Step size for numerical algorithms (default: 1e-9).
         * @param tolerance Convergence criterion for numerical methods (default: 1e-6).
         * @param timeout Maximum iterations for numerical methods (default: 1000).
         * @return A std::map containing torch::Tensor objects representing the kinematics of the two neutrinos,
         *         similar to the `NuNu(H1_, H1_perp, ...)` function.
         */
        std::map<std::string, torch::Tensor> NuNu(
                        torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
                        torch::Tensor* met_xy, double null, torch::Tensor* m1, torch::Tensor* m2 = nullptr,
                        const double step = 1e-9, const double tolerance = 1e-6, const unsigned int timeout = 1000
        );

        /**
         * @brief Solves for di-neutrino kinematics directly from particle momenta using explicit mass values.
         *
         * Similar to the previous `NuNu` overload, this combines base component calculation and
         * di-neutrino solving. However, it uses explicitly provided, fixed mass values for the
         * top quarks (massT1, massT2) and W bosons (massW1, massW2) for each decay leg.
         * Neutrino masses are likely assumed zero internally.
         *
         * @param pmc_b1 Pointer to tensor for the first b-quark four-momenta. Shape: [N, 4].
         * @param pmc_b2 Pointer to tensor for the second b-quark four-momenta. Shape: [N, 4].
         * @param pmc_mu1 Pointer to tensor for the first muon four-momenta. Shape: [N, 4].
         * @param pmc_mu2 Pointer to tensor for the second muon four-momenta. Shape: [N, 4].
         * @param met_xy Pointer to tensor for the total MET vector. Shape: [N, 2].
         * @param null A double value used to indicate failed solutions or non-physical results.
         * @param massT1 Fixed top quark mass for the first decay leg (double).
         * @param massW1 Fixed W boson mass for the first decay leg (double).
         * @param massT2 Fixed top quark mass for the second decay leg (double).
         * @param massW2 Fixed W boson mass for the second decay leg (double).
         * @return A std::map containing torch::Tensor objects representing the kinematics of the two neutrinos,
         *         similar to the `NuNu(H1_, H1_perp, ...)` function. Note that numerical parameters (step, tolerance, timeout)
         *         use internal defaults in this overload.
         */
        std::map<std::string, torch::Tensor> NuNu(
                        torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
                        torch::Tensor* met_xy, double null, double massT1, double massW1, double massT2, double massW2
        );

} // namespace nusol_
#endif // CU_NUSOL_BASE_H
