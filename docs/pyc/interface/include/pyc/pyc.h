#ifndef PYC_H
#define PYC_H

#include <tuple>
#include <torch/all.h>
#include <templates/particle_template.h>

/**
 * @class neutrino
 * @brief Represents a reconstructed neutrino particle, inheriting from a base particle template.
 * @details This class extends the `particle_template` to include additional information relevant
 *          to neutrino reconstruction, such as associated b-quark and lepton indices, pointers
 *          to the associated particles, and potentially a quality metric (`min`).
 *          It's primarily used as a return type for neutrino reconstruction functions.
 */
class neutrino : public particle_template
{
    public:
        /**
         * @brief Inherits constructors from the base `particle_template` class.
         * @details Allows creating `neutrino` objects using the same constructors defined
         *          in `particle_template`, typically initializing the four-momentum.
         */
        using particle_template::particle_template;

        /**
         * @brief Virtual destructor for the neutrino class.
         * @details Ensures proper cleanup when deleting `neutrino` objects, especially
         *          if derived classes were to exist. Also handles potential cleanup
         *          of owned resources like the `bquark` and `lepton` pointers if necessary
         *          (though current implementation seems to imply external ownership or copies).
         */
        virtual ~neutrino();

        /**
         * @brief A double value, potentially representing a quality metric or discriminant.
         * @details Its specific meaning depends on the reconstruction algorithm that produced
         *          this neutrino object. It might store a chi-squared value, a likelihood,
         *          or the minimum value of some objective function. Initialized to 0.
         */
        double min = 0;

        /**
         * @brief Index of the associated lepton in the original input collections.
         * @details Used to link this reconstructed neutrino back to the specific lepton
         *          involved in the decay chain (e.g., W -> l nu). Initialized to -1.
         */
        long l_idx = -1;

        /**
         * @brief Index of the associated b-quark in the original input collections.
         * @details Used to link this reconstructed neutrino back to the specific b-quark
         *          involved in the decay chain (e.g., t -> b W). Initialized to -1.
         */
        long b_idx = -1;

        /**
         * @brief Pointer to the associated b-quark particle object.
         * @details Stores a pointer to a copy or the original `particle_template` (or derived type)
         *          representing the b-quark associated with this neutrino's decay chain.
         *          Ownership details depend on how it's populated by the reconstruction function.
         *          Initialized to nullptr.
         */
        particle_template* bquark = nullptr;

        /**
         * @brief Pointer to the associated lepton particle object.
         * @details Stores a pointer to a copy or the original `particle_template` (or derived type)
         *          representing the lepton associated with this neutrino's decay chain.
         *          Ownership details depend on how it's populated by the reconstruction function.
         *          Initialized to nullptr.
         */
        particle_template* lepton = nullptr;
};

/**
 * @brief Namespace containing utility functions for C++/PyTorch interoperability,
 *        physics calculations, coordinate transformations, and graph operations.
 * @details This namespace provides a collection of tools designed to facilitate
 *          high-energy physics analysis workflows that involve both C++ data structures
 *          and PyTorch tensors. It covers data conversion, kinematic calculations in
 *          different coordinate systems, matrix operations, neutrino reconstruction
 *          algorithms, and graph-based analysis utilities.
 */
namespace pyc {

    /**
     * @brief Converts a pointer to a `std::map<std::string, torch::Tensor>` to a `torch::Dict<std::string, torch::Tensor>`.
     * @details This function facilitates passing C++ maps containing named tensors to PyTorch C++ API functions
     *          that expect `torch::Dict`. The conversion shares ownership of the underlying tensors, meaning
     *          modifications to tensors in the output dictionary will affect the tensors in the original map,
     *          and vice versa. No data is copied.
     * @param inpt Pointer to the input `std::map<std::string, torch::Tensor>`. The map itself is not modified.
     * @return A `torch::Dict<std::string, torch::Tensor>` containing the same key-value pairs as the input map.
     *         Tensor ownership is shared.
     */
    torch::Dict<std::string, torch::Tensor> std_to_dict(std::map<std::string, torch::Tensor>* inpt);

    /**
     * @brief Converts a `std::map<std::string, torch::Tensor>` by value to a `torch::Dict<std::string, torch::Tensor>`.
     * @details This function creates a `torch::Dict` from a C++ `std::map`. Since the map is passed by value,
     *          the tensors within the map are effectively copied into the new dictionary structure.
     *          Modifications to tensors in the output dictionary will *not* affect the tensors in the original map.
     * @param inpt The input `std::map<std::string, torch::Tensor>` (passed by value, potentially involving a copy of the map structure and tensor references).
     * @return A `torch::Dict<std::string, torch::Tensor>` containing copies of the key-value pairs from the input map.
     *         Tensor data is effectively copied.
     */
    torch::Dict<std::string, torch::Tensor> std_to_dict(std::map<std::string, torch::Tensor> inpt);

    /**
     * @brief Converts a pointer to a `std::vector<std::vector<double>>` into a 2D `torch::Tensor` of type `torch::kDouble`.
     * @details Creates a PyTorch tensor from a C++ nested vector representing a matrix or a list of vectors.
     *          The data from the vector is copied into the tensor's memory. Assumes all inner vectors have the same size.
     * @param inpt Pointer to the input `std::vector<std::vector<double>>`. The vector itself is not modified.
     * @return A `torch::Tensor` with shape `[N, M]`, where `N` is the size of the outer vector
     *         and `M` is the size of the inner vectors. The tensor data type is `torch::kDouble`.
     *         The tensor owns its data.
     */
    torch::Tensor tensorize(std::vector<std::vector<double>>* inpt);

    /**
     * @brief Converts a pointer to a `std::vector<std::vector<long>>` into a 2D `torch::Tensor` of type `torch::kLong`.
     * @details Creates a PyTorch tensor from a C++ nested vector representing a matrix or a list of vectors of integers.
     *          The data from the vector is copied into the tensor's memory. Assumes all inner vectors have the same size.
     * @param inpt Pointer to the input `std::vector<std::vector<long>>`. The vector itself is not modified.
     * @return A `torch::Tensor` with shape `[N, M]`, where `N` is the size of the outer vector
     *         and `M` is the size of the inner vectors. The tensor data type is `torch::kLong`.
     *         The tensor owns its data.
     */
    torch::Tensor tensorize(std::vector<std::vector<long>>* inpt);

    /**
     * @brief Converts a pointer to a `std::vector<double>` into a 1D `torch::Tensor` of type `torch::kDouble`.
     * @details Creates a PyTorch tensor from a C++ vector of doubles.
     *          The data from the vector is copied into the tensor's memory.
     * @param inpt Pointer to the input `std::vector<double>`. The vector itself is not modified.
     * @return A `torch::Tensor` with shape `[N]`, where `N` is the size of the vector.
     *         The tensor data type is `torch::kDouble`. The tensor owns its data.
     */
    torch::Tensor tensorize(std::vector<double>* inpt);

    /**
     * @brief Converts a pointer to a `std::vector<long>` into a 1D `torch::Tensor` of type `torch::kLong`.
     * @details Creates a PyTorch tensor from a C++ vector of longs.
     *          The data from the vector is copied into the tensor's memory.
     * @param inpt Pointer to the input `std::vector<long>`. The vector itself is not modified.
     * @return A `torch::Tensor` with shape `[N]`, where `N` is the size of the vector.
     *         The tensor data type is `torch::kLong`. The tensor owns its data.
     */
    torch::Tensor tensorize(std::vector<long>* inpt);

    /**
     * @brief Template function to extract Cartesian four-momentum (Px, Py, Pz, E) from a generic particle object.
     * @details This utility function provides a standard way to get the four-momentum components
     *          from any particle-like object that exposes `px`, `py`, `pz`, and `e` members (or methods).
     *          It returns these components in a `std::vector<double>`.
     * @tparam g The type of the particle object. Must have accessible members or methods named `px`, `py`, `pz`, and `e` that return numerical types convertible to double.
     * @param p Pointer to the particle object. The object itself is not modified.
     * @return A `std::vector<double>` containing `{px, py, pz, e}`.
     */
    template <typename g>
    std::vector<double> as_pmc(g* p);

    /**
     * @brief Template function to convert a vector of pointers to generic particle objects into a vector of Cartesian four-momenta.
     * @details Iterates through a vector of particle pointers, calls `as_pmc` on each particle,
     *          and collects the resulting four-momentum vectors into a nested `std::vector`.
     *          This is useful for preparing particle data for functions that expect kinematics as `std::vector<std::vector<double>>`.
     * @tparam g The type of the particle objects in the input vector. Must be compatible with `as_pmc<g>`.
     * @param p Pointer to a `std::vector` of pointers to particle objects (`std::vector<g*>*`). The input vector and its elements are not modified.
     * @return A `std::vector<std::vector<double>>`, where each inner vector is the `{px, py, pz, e}` representation
     *         obtained by calling `as_pmc` on each element of the input vector. The size of the outer vector matches the size of the input vector.
     */
    template <typename g>
    std::vector<std::vector<double>> to_pmc(std::vector<g*>* p);

    /**
     * @brief Namespace containing functions for coordinate transformations between Cartesian (Px, Py, Pz, E)
     *        and Polar (Pt, Eta, Phi, E/Mass) systems for particle kinematics.
     * @details Provides tools to switch between different representations of four-momenta,
     *          which are commonly used in high-energy physics analyses. Offers versions that
     *          operate on separate component tensors and versions that operate on combined tensors.
     */
    namespace transform {
        /**
         * @brief Namespace for transformations where kinematic components (Px, Py, Pz, E or Pt, Eta, Phi, E)
         *        are provided as separate `torch::Tensor` objects.
         * @details Functions in this namespace expect each kinematic variable (e.g., `px`, `py`) as an individual tensor argument.
         *          All input tensors for a single function call are expected to have compatible shapes (usually 1D of the same size `N`).
         */
        namespace separate {
            /**
             * @brief Calculates the transverse momentum (Pt) from Cartesian components Px and Py.
             * @details Computes `sqrt(px^2 + py^2)` element-wise.
             * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
             * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
             * @return Tensor containing the transverse momentum. Shape: `[N]`. Data type matches input tensors.
             */
            torch::Tensor Pt(torch::Tensor px, torch::Tensor py);

            /**
             * @brief Calculates the pseudorapidity (Eta) from Cartesian components Px, Py, and Pz.
             * @details Computes `asinh(pz / Pt)` or equivalently `-ln(tan(theta/2))`, where `Pt = sqrt(px^2 + py^2)` and `theta` is the polar angle.
             *          Handles cases where `Pt` is zero (typically returns 0 or +/- infinity depending on `pz`).
             * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
             * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
             * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
             * @return Tensor containing the pseudorapidity. Shape: `[N]`. Data type matches input tensors.
             */
            torch::Tensor Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

            /**
             * @brief Calculates the azimuthal angle (Phi) from Cartesian components Px and Py.
             * @details Computes `atan2(py, px)` element-wise. The result is typically in the range `[-pi, pi]`.
             * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
             * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
             * @return Tensor containing the azimuthal angle in radians. Shape: `[N]`. Data type matches input tensors.
             */
            torch::Tensor Phi(torch::Tensor px, torch::Tensor py);

            /**
             * @brief Calculates the polar coordinates (Pt, Eta, Phi) from Cartesian components Px, Py, and Pz.
             * @details Combines the calculations of `Pt(px, py)`, `Eta(px, py, pz)`, and `Phi(px, py)`.
             * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
             * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
             * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
             * @return Tensor of shape `[N, 3]` containing `(Pt, Eta, Phi)` columns for each input element. Data type matches input tensors.
             */
            torch::Tensor PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

            /**
             * @brief Calculates the polar coordinates including energy (Pt, Eta, Phi, E) from Cartesian components Px, Py, Pz, and E.
             * @details Combines the calculations of `Pt(px, py)`, `Eta(px, py, pz)`, `Phi(px, py)`, and includes the original energy `e`.
             * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
             * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
             * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
             * @param e Tensor containing the energy. Shape: `[N]`.
             * @return Tensor of shape `[N, 4]` containing `(Pt, Eta, Phi, E)` columns for each input element. Data type matches input tensors.
             */
            torch::Tensor PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

            /**
             * @brief Calculates the Cartesian x-component (Px) from polar coordinates Pt and Phi.
             * @details Computes `pt * cos(phi)` element-wise.
             * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
             * @param phi Tensor containing the azimuthal angle in radians. Shape: `[N]`.
             * @return Tensor containing the x-component of momentum. Shape: `[N]`. Data type matches input tensors.
             */
            torch::Tensor Px(torch::Tensor pt, torch::Tensor phi);

            /**
             * @brief Calculates the Cartesian y-component (Py) from polar coordinates Pt and Phi.
             * @details Computes `pt * sin(phi)` element-wise.
             * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
             * @param phi Tensor containing the azimuthal angle in radians. Shape: `[N]`.
             * @return Tensor containing the y-component of momentum. Shape: `[N]`. Data type matches input tensors.
             */
            torch::Tensor Py(torch::Tensor pt, torch::Tensor phi);

            /**
             * @brief Calculates the Cartesian z-component (Pz) from polar coordinates Pt and Eta.
             * @details Computes `pt * sinh(eta)` element-wise.
             * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
             * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
             * @return Tensor containing the z-component of momentum. Shape: `[N]`. Data type matches input tensors.
             */
            torch::Tensor Pz(torch::Tensor pt, torch::Tensor eta);

            /**
             * @brief Calculates the Cartesian momentum components (Px, Py, Pz) from polar coordinates Pt, Eta, and Phi.
             * @details Combines the calculations of `Px(pt, phi)`, `Py(pt, phi)`, and `Pz(pt, eta)`.
             * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
             * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
             * @param phi Tensor containing the azimuthal angle in radians. Shape: `[N]`.
             * @return Tensor of shape `[N, 3]` containing `(Px, Py, Pz)` columns for each input element. Data type matches input tensors.
             */
            torch::Tensor PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);

            /**
             * @brief Calculates the Cartesian four-momentum (Px, Py, Pz, E) from polar coordinates Pt, Eta, Phi, and E.
             * @details Combines the calculations of `Px(pt, phi)`, `Py(pt, phi)`, `Pz(pt, eta)`, and includes the original energy `e`.
             * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
             * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
             * @param phi Tensor containing the azimuthal angle in radians. Shape: `[N]`.
             * @param e Tensor containing the energy. Shape: `[N]`.
             * @return Tensor of shape `[N, 4]` containing `(Px, Py, Pz, E)` columns for each input element. Data type matches input tensors.
             */
            torch::Tensor PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);
        } // namespace separate

        /**
         * @brief Namespace for transformations where kinematic components are provided as a single combined tensor.
         * @details Functions in this namespace expect kinematic variables grouped into columns of a single tensor.
         *          For Cartesian input (`pmc`), the expected column order is `(Px, Py, Pz, E)`.
         *          For Polar input (`pmu`), the expected column order is `(Pt, Eta, Phi, E)` or `(Pt, Eta, Phi, Mass)`.
         *          Input tensors typically have shape `[N, 4]`.
         */
        namespace combined {
            /**
             * @brief Calculates the transverse momentum (Pt) from a combined Cartesian four-momentum tensor.
             * @details Extracts Px (column 0) and Py (column 1) and computes `sqrt(Px^2 + Py^2)`.
             * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
             * @return Tensor containing the transverse momentum. Shape: `[N]`.
             */
            torch::Tensor Pt(torch::Tensor pmc);

            /**
             * @brief Calculates the pseudorapidity (Eta) from a combined Cartesian four-momentum tensor.
             * @details Extracts Px (col 0), Py (col 1), Pz (col 2) and computes `asinh(Pz / Pt)`.
             * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
             * @return Tensor containing the pseudorapidity. Shape: `[N]`.
             */
            torch::Tensor Eta(torch::Tensor pmc);

            /**
             * @brief Calculates the azimuthal angle (Phi) from a combined Cartesian four-momentum tensor.
             * @details Extracts Px (col 0), Py (col 1) and computes `atan2(Py, Px)`.
             * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
             * @return Tensor containing the azimuthal angle in radians. Shape: `[N]`.
             */
            torch::Tensor Phi(torch::Tensor pmc);

            /**
             * @brief Calculates the polar coordinates (Pt, Eta, Phi) from a combined Cartesian four-momentum tensor.
             * @details Extracts Px, Py, Pz and computes Pt, Eta, Phi, stacking them column-wise.
             * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
             * @return Tensor of shape `[N, 3]` containing `(Pt, Eta, Phi)`.
             */
            torch::Tensor PtEtaPhi(torch::Tensor pmc);

            /**
             * @brief Calculates the polar coordinates including energy (Pt, Eta, Phi, E) from a combined Cartesian four-momentum tensor.
             * @details Extracts Px, Py, Pz, E and computes Pt, Eta, Phi, stacking them with the original E column-wise.
             * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
             * @return Tensor of shape `[N, 4]` containing `(Pt, Eta, Phi, E)`.
             */
            torch::Tensor PtEtaPhiE(torch::Tensor pmc);

            /**
             * @brief Calculates the Cartesian x-component (Px) from a combined polar coordinate tensor.
             * @details Extracts Pt (column 0) and Phi (column 2) and computes `Pt * cos(Phi)`.
             * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E or M)`.
             * @return Tensor containing the x-component of momentum. Shape: `[N]`.
             */
            torch::Tensor Px(torch::Tensor pmu);

            /**
             * @brief Calculates the Cartesian y-component (Py) from a combined polar coordinate tensor.
             * @details Extracts Pt (column 0) and Phi (column 2) and computes `Pt * sin(Phi)`.
             * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E or M)`.
             * @return Tensor containing the y-component of momentum. Shape: `[N]`.
             */
            torch::Tensor Py(torch::Tensor pmu);

            /**
             * @brief Calculates the Cartesian z-component (Pz) from a combined polar coordinate tensor.
             * @details Extracts Pt (column 0) and Eta (column 1) and computes `Pt * sinh(Eta)`.
             * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E or M)`.
             * @return Tensor containing the z-component of momentum. Shape: `[N]`.
             */
            torch::Tensor Pz(torch::Tensor pmu);

            /**
             * @brief Calculates the Cartesian momentum components (Px, Py, Pz) from a combined polar coordinate tensor.
             * @details Extracts Pt (col 0), Eta (col 1), Phi (col 2) and computes Px, Py, Pz, stacking them column-wise.
             * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E or M)`.
             * @return Tensor of shape `[N, 3]` containing `(Px, Py, Pz)`.
             */
            torch::Tensor PxPyPz(torch::Tensor pmu);

            /**
             * @brief Calculates the Cartesian four-momentum (Px, Py, Pz, E) from a combined polar coordinate tensor.
             * @details Extracts Pt (col 0), Eta (col 1), Phi (col 2), E (col 3) and computes Px, Py, Pz, stacking them with the original E column-wise.
             * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E)`. Assumes the fourth column is Energy.
             * @return Tensor of shape `[N, 4]` containing `(Px, Py, Pz, E)`.
             */
            torch::Tensor PxPyPzE(torch::Tensor pmu);
        } // namespace combined
    } // namespace transform

    /**
     * @brief Namespace containing functions for calculating common relativistic physics quantities.
     * @details Provides functions to compute invariant mass, momentum magnitude, beta, transverse mass,
     *          angles, and angular separations using either Cartesian or Polar coordinates.
     *          Offers versions operating on separate component tensors and combined tensors.
     */
    namespace physics {
        /**
         * @brief Namespace for physics calculations using Cartesian coordinates (Px, Py, Pz, E).
         */
        namespace cartesian {
            /**
             * @brief Namespace for calculations where Cartesian components (Px, Py, Pz, E) are provided as separate tensors.
             * @details Functions expect individual tensors for each component, typically of shape `[N]`.
             */
            namespace separate {
                /**
                 * @brief Calculates the squared magnitude of the 3-momentum (P^2).
                 * @details Computes `px^2 + py^2 + pz^2` element-wise.
                 * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
                 * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
                 * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
                 * @return Tensor containing `P^2`. Shape: `[N]`.
                 */
                torch::Tensor P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

                /**
                 * @brief Calculates the magnitude of the 3-momentum (P).
                 * @details Computes `sqrt(px^2 + py^2 + pz^2)` element-wise. Equivalent to `sqrt(P2(px, py, pz))`.
                 * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
                 * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
                 * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
                 * @return Tensor containing `P`. Shape: `[N]`.
                 */
                torch::Tensor P(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

                /**
                 * @brief Calculates the squared relativistic beta (beta^2 = (v/c)^2).
                 * @details Computes `P^2 / e^2`, where `P^2 = px^2 + py^2 + pz^2`.
                 *          Handles division by zero by returning 0 where `e` is 0.
                 * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
                 * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
                 * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `beta^2`. Shape: `[N]`. Values are between 0 and 1 (or slightly > 1 due to numerical precision for massless particles).
                 */
                torch::Tensor Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

                /**
                 * @brief Calculates the relativistic beta (beta = v/c).
                 * @details Computes `P / e`, where `P = sqrt(px^2 + py^2 + pz^2)`.
                 *          Handles division by zero by returning 0 where `e` is 0.
                 * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
                 * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
                 * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `beta`. Shape: `[N]`. Values are between 0 and 1.
                 */
                torch::Tensor Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

                /**
                 * @brief Calculates the squared invariant mass (M^2).
                 * @details Computes `e^2 - P^2 = e^2 - (px^2 + py^2 + pz^2)` element-wise.
                 *          The result is clamped at 0 to avoid negative values due to numerical precision.
                 * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
                 * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
                 * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `M^2`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

                /**
                 * @brief Calculates the invariant mass (M).
                 * @details Computes `sqrt(M2(px, py, pz, e))`. Ensures non-negativity by calculating from `M2`.
                 * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
                 * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
                 * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `M`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

                /**
                 * @brief Calculates the squared transverse mass (Mt^2), defined here as `e^2 - pz^2`.
                 * @details Computes `e^2 - pz^2` element-wise. Clamped at 0.
                 * @note This definition `Mt^2 = e^2 - pz^2` differs from the more common definition involving Pt, e.g., `Mt^2 = Et^2 - pz^2` (where `Et^2 = m^2 + pt^2`) or `Mt^2 = m_lepton^2 + m_nu^2 + 2(Et_l Et_nu - pt_l . pt_nu)`. Verify the intended definition based on usage context.
                 * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `Mt^2 = e^2 - pz^2`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor Mt2(torch::Tensor pz, torch::Tensor e);

                /**
                 * @brief Calculates the transverse mass (Mt), defined here as `sqrt(e^2 - pz^2)`.
                 * @details Computes `sqrt(Mt2(pz, e))`.
                 * @note This definition `Mt = sqrt(e^2 - pz^2)` is derived from the `Mt2` definition above and may differ from common usage.
                 * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `Mt = sqrt(e^2 - pz^2)`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor Mt(torch::Tensor pz, torch::Tensor e);

                /**
                 * @brief Calculates the polar angle (Theta) with respect to the positive z-axis.
                 * @details Computes `atan2(Pt, pz)`, where `Pt = sqrt(px^2 + py^2)`. Result is in radians, typically in `[0, pi]`.
                 * @param px Tensor containing the x-components of momentum. Shape: `[N]`.
                 * @param py Tensor containing the y-components of momentum. Shape: `[N]`.
                 * @param pz Tensor containing the z-components of momentum. Shape: `[N]`.
                 * @return Tensor containing the polar angle `Theta` in radians. Shape: `[N]`.
                 */
                torch::Tensor Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

                /**
                 * @brief Calculates the angular separation Delta R = sqrt(DeltaEta^2 + DeltaPhi^2) between two particles using Cartesian coordinates.
                 * @details First calculates `eta1`, `phi1` from `(px1, py1, pz1)` and `eta2`, `phi2` from `(px2, py2, pz2)`.
                 *          Then computes `delta_eta = eta1 - eta2` and `delta_phi = phi1 - phi2` (handling the `[-pi, pi]` wrap-around for `delta_phi`).
                 *          Finally, returns `sqrt(delta_eta^2 + delta_phi^2)`.
                 * @param px1 Tensor for Px of the first particle set. Shape: `[N]`.
                 * @param px2 Tensor for Px of the second particle set. Shape: `[N]`.
                 * @param py1 Tensor for Py of the first particle set. Shape: `[N]`.
                 * @param py2 Tensor for Py of the second particle set. Shape: `[N]`.
                 * @param pz1 Tensor for Pz of the first particle set. Shape: `[N]`.
                 * @param pz2 Tensor for Pz of the second particle set. Shape: `[N]`.
                 * @return Tensor containing the Delta R values. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor DeltaR(torch::Tensor px1, torch::Tensor px2, torch::Tensor py1, torch::Tensor py2, torch::Tensor pz1, torch::Tensor pz2);
            } // namespace separate

            /**
             * @brief Namespace for calculations where Cartesian components are provided as a single combined tensor `pmc = (Px, Py, Pz, E)`.
             * @details Functions expect input tensors of shape `[N, 4]`.
             */
            namespace combined {
                /**
                 * @brief Calculates the squared magnitude of the 3-momentum (P^2) from a combined Cartesian tensor.
                 * @details Extracts Px (col 0), Py (col 1), Pz (col 2) and computes `Px^2 + Py^2 + Pz^2`.
                 * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
                 * @return Tensor containing `P^2`. Shape: `[N]`.
                 */
                torch::Tensor P2(torch::Tensor pmc);

                /**
                 * @brief Calculates the magnitude of the 3-momentum (P) from a combined Cartesian tensor.
                 * @details Extracts Px, Py, Pz and computes `sqrt(Px^2 + Py^2 + Pz^2)`. Equivalent to `sqrt(P2(pmc))`.
                 * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
                 * @return Tensor containing `P`. Shape: `[N]`.
                 */
                torch::Tensor P(torch::Tensor pmc);

                /**
                 * @brief Calculates the squared relativistic beta (beta^2) from a combined Cartesian tensor.
                 * @details Extracts Px, Py, Pz, E (col 3) and computes `P2(pmc) / E^2`. Handles division by zero.
                 * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
                 * @return Tensor containing `beta^2`. Shape: `[N]`.
                 */
                torch::Tensor Beta2(torch::Tensor pmc);

                /**
                 * @brief Calculates the relativistic beta (beta) from a combined Cartesian tensor.
                 * @details Extracts Px, Py, Pz, E and computes `P(pmc) / E`. Handles division by zero.
                 * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
                 * @return Tensor containing `beta`. Shape: `[N]`.
                 */
                torch::Tensor Beta(torch::Tensor pmc);

                /**
                 * @brief Calculates the squared invariant mass (M^2) from a combined Cartesian tensor.
                 * @details Extracts Px, Py, Pz, E and computes `E^2 - P2(pmc)`. Clamped at 0.
                 * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
                 * @return Tensor containing `M^2`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor M2(torch::Tensor pmc);

                /**
                 * @brief Calculates the invariant mass (M) from a combined Cartesian tensor.
                 * @details Computes `sqrt(M2(pmc))`. Ensures non-negativity.
                 * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
                 * @return Tensor containing `M`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor M(torch::Tensor pmc);

                /**
                 * @brief Calculates the squared transverse mass (Mt^2), defined here as `E^2 - Pz^2`, from a combined Cartesian tensor.
                 * @details Extracts Pz (col 2) and E (col 3) and computes `E^2 - Pz^2`. Clamped at 0.
                 * @note See note in `physics::cartesian::separate::Mt2` regarding the definition.
                 * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
                 * @return Tensor containing `Mt^2 = E^2 - Pz^2`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor Mt2(torch::Tensor pmc);

                /**
                 * @brief Calculates the transverse mass (Mt), defined here as `sqrt(E^2 - Pz^2)`, from a combined Cartesian tensor.
                 * @details Computes `sqrt(Mt2(pmc))`.
                 * @note See note in `physics::cartesian::separate::Mt` regarding the definition.
                 * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
                 * @return Tensor containing `Mt = sqrt(E^2 - Pz^2)`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor Mt(torch::Tensor pmc);

                /**
                 * @brief Calculates the polar angle (Theta) with respect to the z-axis from a combined Cartesian tensor.
                 * @details Extracts Px, Py, Pz and computes `atan2(Pt, Pz)`, where `Pt = sqrt(Px^2 + Py^2)`.
                 * @param pmc Tensor of shape `[N, 4]` representing `(Px, Py, Pz, E)`.
                 * @return Tensor containing the polar angle `Theta` in radians. Shape: `[N]`.
                 */
                torch::Tensor Theta(torch::Tensor pmc);

                /**
                 * @brief Calculates the angular separation Delta R between two particles using combined Cartesian tensors.
                 * @details Calculates `eta1`, `phi1` from `pmc1` and `eta2`, `phi2` from `pmc2`.
                 *          Then computes `delta_eta = eta1 - eta2` and `delta_phi = phi1 - phi2` (handling wrap-around).
                 *          Finally, returns `sqrt(delta_eta^2 + delta_phi^2)`.
                 * @param pmc1 Tensor of shape `[N, 4]` for the first particle set `(Px, Py, Pz, E)`.
                 * @param pmc2 Tensor of shape `[N, 4]` for the second particle set `(Px, Py, Pz, E)`.
                 * @return Tensor containing the Delta R values. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor DeltaR(torch::Tensor pmc1, torch::Tensor pmc2);
            } // namespace combined
        } // namespace cartesian

        /**
         * @brief Namespace for physics calculations using Polar coordinates (Pt, Eta, Phi, E/M).
         */
        namespace polar {
            /**
             * @brief Namespace for calculations where Polar components (Pt, Eta, Phi, E/M) are provided as separate tensors.
             * @details Functions expect individual tensors for each component, typically of shape `[N]`.
             *          The `phi` component is often unused in calculations like P, M, Beta but included for signature consistency.
             */
            namespace separate {
                /**
                 * @brief Calculates the squared magnitude of the 3-momentum (P^2) from polar coordinates.
                 * @details Computes `Pz^2 + Pt^2 = (pt * sinh(eta))^2 + pt^2 = pt^2 * (sinh^2(eta) + 1) = pt^2 * cosh^2(eta)`.
                 * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
                 * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
                 * @param phi Tensor containing the azimuthal angle (unused in calculation). Shape: `[N]`.
                 * @return Tensor containing `P^2`. Shape: `[N]`.
                 */
                torch::Tensor P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);

                /**
                 * @brief Calculates the magnitude of the 3-momentum (P) from polar coordinates.
                 * @details Computes `sqrt(P2(pt, eta, phi))` which simplifies to `abs(pt * cosh(eta))`. Since Pt >= 0 and cosh(eta) >= 1, this is `pt * cosh(eta)`.
                 * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
                 * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
                 * @param phi Tensor containing the azimuthal angle (unused in calculation). Shape: `[N]`.
                 * @return Tensor containing `P`. Shape: `[N]`.
                 */
                torch::Tensor P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);

                /**
                 * @brief Calculates the squared relativistic beta (beta^2) from polar coordinates and energy.
                 * @details Computes `P2(pt, eta, phi) / e^2`. Handles division by zero.
                 * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
                 * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
                 * @param phi Tensor containing the azimuthal angle (unused in calculation). Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `beta^2`. Shape: `[N]`.
                 */
                torch::Tensor Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);

                /**
                 * @brief Calculates the relativistic beta (beta) from polar coordinates and energy.
                 * @details Computes `P(pt, eta, phi) / e`. Handles division by zero.
                 * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
                 * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
                 * @param phi Tensor containing the azimuthal angle (unused in calculation). Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `beta`. Shape: `[N]`.
                 */
                torch::Tensor Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);

                /**
                 * @brief Calculates the squared invariant mass (M^2) from polar coordinates and energy.
                 * @details Computes `e^2 - P2(pt, eta, phi)`. Clamped at 0.
                 * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
                 * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
                 * @param phi Tensor containing the azimuthal angle (unused in calculation). Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `M^2`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);

                /**
                 * @brief Calculates the invariant mass (M) from polar coordinates and energy.
                 * @details Computes `sqrt(M2(pt, eta, phi, e))`. Ensures non-negativity.
                 * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
                 * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
                 * @param phi Tensor containing the azimuthal angle (unused in calculation). Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `M`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);

                /**
                 * @brief Calculates the squared transverse mass (Mt^2), defined here as `e^2 - pz^2`, using polar inputs.
                 * @details Computes `e^2 - (pt * sinh(eta))^2`. Clamped at 0.
                 * @note See note in `physics::cartesian::separate::Mt2` regarding the definition.
                 * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
                 * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `Mt^2 = e^2 - pz^2`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e);

                /**
                 * @brief Calculates the transverse mass (Mt), defined here as `sqrt(e^2 - pz^2)`, using polar inputs.
                 * @details Computes `sqrt(Mt2(pt, eta, e))`.
                 * @note See note in `physics::cartesian::separate::Mt` regarding the definition.
                 * @param pt Tensor containing the transverse momentum. Shape: `[N]`.
                 * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
                 * @param e Tensor containing the energy. Shape: `[N]`.
                 * @return Tensor containing `Mt = sqrt(e^2 - pz^2)`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e);

                /**
                 * @brief Calculates the polar angle (Theta) with respect to the z-axis from polar coordinates.
                 * @details Computes `2 * atan(exp(-eta))`. This is equivalent to `atan2(Pt, Pz)` but uses only eta.
                 * @param pt Tensor containing the transverse momentum (unused, but kept for consistency). Shape: `[N]`.
                 * @param eta Tensor containing the pseudorapidity. Shape: `[N]`.
                 * @param phi Tensor containing the azimuthal angle (unused). Shape: `[N]`.
                 * @return Tensor containing `Theta` in radians. Shape: `[N]`. Values are in `[0, pi]`.
                 */
                torch::Tensor Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);

                /**
                 * @brief Calculates the angular separation Delta R = sqrt(DeltaEta^2 + DeltaPhi^2) between two particles using polar coordinates.
                 * @details Computes `delta_eta = eta1 - eta2` and `delta_phi = phi1 - phi2` (handling the `[-pi, pi]` wrap-around for `delta_phi`).
                 *          Returns `sqrt(delta_eta^2 + delta_phi^2)`. This is more direct than the Cartesian version if inputs are already in polar coordinates.
                 * @param eta1 Tensor for Eta of the first particle set. Shape: `[N]`.
                 * @param eta2 Tensor for Eta of the second particle set. Shape: `[N]`.
                 * @param phi1 Tensor for Phi of the first particle set (in radians). Shape: `[N]`.
                 * @param phi2 Tensor for Phi of the second particle set (in radians). Shape: `[N]`.
                 * @return Tensor containing the Delta R values. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2);
            } // namespace separate

            /**
             * @brief Namespace for calculations where Polar components are provided as a single combined tensor `pmu = (Pt, Eta, Phi, E or M)`.
             * @details Functions expect input tensors of shape `[N, 4]`. The interpretation of the 4th column (Energy or Mass) depends on the specific function.
             */
            namespace combined {
                /**
                 * @brief Calculates the squared magnitude of the 3-momentum (P^2) from a combined polar tensor.
                 * @details Extracts Pt (col 0) and Eta (col 1) and computes `Pt^2 * cosh^2(Eta)`.
                 * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E or M)`.
                 * @return Tensor containing `P^2`. Shape: `[N]`.
                 */
                torch::Tensor P2(torch::Tensor pmu);

                /**
                 * @brief Calculates the magnitude of the 3-momentum (P) from a combined polar tensor.
                 * @details Extracts Pt (col 0) and Eta (col 1) and computes `Pt * cosh(Eta)`. Equivalent to `sqrt(P2(pmu))`.
                 * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E or M)`.
                 * @return Tensor containing `P`. Shape: `[N]`.
                 */
                torch::Tensor P(torch::Tensor pmu);

                /**
                 * @brief Calculates the squared relativistic beta (beta^2) from a combined polar tensor.
                 * @details Extracts Pt, Eta, E (col 3) and computes `P2(pmu) / E^2`. Assumes 4th column is Energy. Handles division by zero.
                 * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E)`.
                 * @return Tensor containing `beta^2`. Shape: `[N]`.
                 */
                torch::Tensor Beta2(torch::Tensor pmu);

                /**
                 * @brief Calculates the relativistic beta (beta) from a combined polar tensor.
                 * @details Extracts Pt, Eta, E (col 3) and computes `P(pmu) / E`. Assumes 4th column is Energy. Handles division by zero.
                 * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E)`.
                 * @return Tensor containing `beta`. Shape: `[N]`.
                 */
                torch::Tensor Beta(torch::Tensor pmu);

                /**
                 * @brief Calculates the squared invariant mass (M^2) from a combined polar tensor.
                 * @details Extracts Pt, Eta, E (col 3) and computes `E^2 - P2(pmu)`. Assumes 4th column is Energy. Clamped at 0.
                 * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E)`.
                 * @return Tensor containing `M^2`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor M2(torch::Tensor pmu);

                /**
                 * @brief Calculates the invariant mass (M) from a combined polar tensor.
                 * @details Computes `sqrt(M2(pmu))`. Assumes 4th column is Energy. Ensures non-negativity.
                 * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E)`.
                 * @return Tensor containing `M`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor M(torch::Tensor pmu);

                /**
                 * @brief Calculates the squared transverse mass (Mt^2), defined here as `E^2 - Pz^2`, from a combined polar tensor.
                 * @details Extracts Pt (col 0), Eta (col 1), E (col 3) and computes `E^2 - (Pt * sinh(Eta))^2`. Assumes 4th column is Energy. Clamped at 0.
                 * @note See note in `physics::cartesian::separate::Mt2` regarding the definition.
                 * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E)`.
                 * @return Tensor containing `Mt^2 = E^2 - Pz^2`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor Mt2(torch::Tensor pmu);

                /**
                 * @brief Calculates the transverse mass (Mt), defined here as `sqrt(E^2 - Pz^2)`, from a combined polar tensor.
                 * @details Computes `sqrt(Mt2(pmu))`. Assumes 4th column is Energy.
                 * @note See note in `physics::cartesian::separate::Mt` regarding the definition.
                 * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E)`.
                 * @return Tensor containing `Mt = sqrt(E^2 - Pz^2)`. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor Mt(torch::Tensor pmu);

                /**
                 * @brief Calculates the polar angle (Theta) with respect to the z-axis from a combined polar tensor.
                 * @details Extracts Eta (col 1) and computes `2 * atan(exp(-Eta))`.
                 * @param pmu Tensor of shape `[N, 4]` representing `(Pt, Eta, Phi, E or M)`.
                 * @return Tensor containing `Theta` in radians. Shape: `[N]`.
                 */
                torch::Tensor Theta(torch::Tensor pmu);

                /**
                 * @brief Calculates the angular separation Delta R between two particles using combined polar tensors.
                 * @details Extracts Eta (col 1) and Phi (col 2) from both `pmu1` and `pmu2`.
                 *          Computes `delta_eta = eta1 - eta2` and `delta_phi = phi1 - phi2` (handling wrap-around).
                 *          Returns `sqrt(delta_eta^2 + delta_phi^2)`.
                 * @param pmu1 Tensor of shape `[N, 4]` for the first particle set `(Pt, Eta, Phi, E/M)`.
                 * @param pmu2 Tensor of shape `[N, 4]` for the second particle set `(Pt, Eta, Phi, E/M)`.
                 * @return Tensor containing the Delta R values. Shape: `[N]`. Values are non-negative.
                 */
                torch::Tensor DeltaR(torch::Tensor pmu1, torch::Tensor pmu2);
            } // namespace combined
        } // namespace polar
    } // namespace physics

    /**
     * @brief Namespace containing various mathematical operators, primarily focused on vector and matrix algebra using `torch::Tensor`.
     * @details Provides functions for dot products, cross products, angles between vectors, rotation matrices,
     *          matrix inversion, determinant calculation, eigenvalue decomposition, and potentially Lorentz transformations.
     *          These operations typically act element-wise on batches of vectors or matrices.
     */
    namespace operators {
        /**
         * @brief Calculates the dot product of two sets of 3-vectors.
         * @details Computes `v1_x*v2_x + v1_y*v2_y + v1_z*v2_z` for each pair of vectors in the batch.
         *          Only the first three components (indices 0, 1, 2) of the input tensors are used, even if they have 4 columns.
         * @param v1 Tensor of shape `[N, 3]` or `[N, 4]` representing the first set of vectors.
         * @param v2 Tensor of shape `[N, 3]` or `[N, 4]` representing the second set of vectors.
         * @return Tensor of shape `[N]` containing the scalar dot products.
         */
        torch::Tensor Dot(torch::Tensor v1, torch::Tensor v2);

        /**
         * @brief Calculates the cosine of the angle between two sets of 3-vectors.
         * @details Computes `(v1 . v2) / (|v1| * |v2|)`, where `.` denotes the dot product and `|v|` is the magnitude of the 3-vector.
         *          Uses only the first three components. Handles cases where either magnitude is zero by returning 0.
         * @param v1 Tensor of shape `[N, 3]` or `[N, 4]` representing the first set of vectors.
         * @param v2 Tensor of shape `[N, 3]` or `[N, 4]` representing the second set of vectors.
         * @return Tensor of shape `[N]` containing the cosine of the angle between corresponding vectors. Values are in `[-1, 1]`.
         */
        torch::Tensor CosTheta(torch::Tensor v1, torch::Tensor v2);

        /**
         * @brief Calculates the sine of the angle between two sets of 3-vectors.
         * @details Computes `|v1 x v2| / (|v1| * |v2|)`, where `x` denotes the cross product and `|v|` is the magnitude.
         *          Uses only the first three components. Handles cases where either magnitude is zero by returning 0.
         * @param v1 Tensor of shape `[N, 3]` or `[N, 4]` representing the first set of vectors.
         * @param v2 Tensor of shape `[N, 3]` or `[N, 4]` representing the second set of vectors.
         * @return Tensor of shape `[N]` containing the sine of the angle between corresponding vectors. Values are in `[0, 1]`.
         */
        torch::Tensor SinTheta(torch::Tensor v1, torch::Tensor v2);

        /**
         * @brief Generates a batch of 3x3 rotation matrices for rotation around the x-axis.
         * @details Creates matrices R_x(angle) = [[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]] for each angle in the input tensor.
         * @param angle Tensor of shape `[N]` containing the rotation angles in radians.
         * @return Tensor of shape `[N, 3, 3]` containing the rotation matrices.
         */
        torch::Tensor Rx(torch::Tensor angle);

        /**
         * @brief Generates a batch of 3x3 rotation matrices for rotation around the y-axis.
         * @details Creates matrices R_y(angle) = [[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]] for each angle in the input tensor.
         * @param angle Tensor of shape `[N]` containing the rotation angles in radians.
         * @return Tensor of shape `[N, 3, 3]` containing the rotation matrices.
         */
        torch::Tensor Ry(torch::Tensor angle);

        /**
         * @brief Generates a batch of 3x3 rotation matrices for rotation around the z-axis.
         * @details Creates matrices R_z(angle) = [[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]] for each angle in the input tensor.
         * @param angle Tensor of shape `[N]` containing the rotation angles in radians.
         * @return Tensor of shape `[N, 3, 3]` containing the rotation matrices.
         */
        torch::Tensor Rz(torch::Tensor angle);

        /**
         * @brief Applies a rotation and potentially a boost (Lorentz transformation) to a set of particles.
         * @details This function likely transforms the four-momenta `pmc_mu` into the rest frame of the particles represented by `pmc_b`,
         *          or applies a similar frame transformation based on the kinematics of `pmc_b`. The exact transformation
         *          (e.g., boost to rest frame, rotation to align axes) depends on the specific implementation.
         *          It typically involves calculating boost vectors and rotation matrices from `pmc_b` and applying them to `pmc_mu`.
         * @param pmc_b Tensor of shape `[N, 4]` representing the reference frame particles (e.g., parent particle, jet axis). Input: `(Px, Py, Pz, E)`.
         * @param pmc_mu Tensor of shape `[N, 4]` representing the particles to be transformed. Input: `(Px, Py, Pz, E)`.
         * @return Tensor of shape `[N, 4]` containing the transformed four-momenta of `pmc_mu`. Output: `(Px', Py', Pz', E')`.
         */
        torch::Tensor RT(torch::Tensor pmc_b, torch::Tensor pmc_mu);

        /**
         * @brief Calculates the cofactor matrix for a batch of square matrices.
         * @details The cofactor C_ij of a matrix A is (-1)^(i+j) times the determinant of the submatrix obtained by removing the i-th row and j-th column of A.
         *          The cofactor matrix is the matrix of cofactors. This is often used in calculating the inverse of a matrix.
         * @param matrix Tensor of shape `[N, D, D]` representing a batch of square matrices.
         * @return Tensor of shape `[N, D, D]` containing the corresponding cofactor matrices.
         */
        torch::Tensor CoFactors(torch::Tensor matrix);

        /**
         * @brief Calculates the determinant for a batch of square matrices.
         * @details Computes the determinant for each `[D, D]` matrix in the batch. Can use cofactor expansion or other methods (e.g., LU decomposition).
         * @param matrix Tensor of shape `[N, D, D]` representing a batch of square matrices.
         * @return Tensor of shape `[N]` containing the determinants.
         */
        torch::Tensor Determinant(torch::Tensor matrix);

        /**
         * @brief Calculates the inverse and determinant for a batch of square matrices.
         * @details Computes the inverse `A^-1` and determinant `det(A)` for each `[D, D]` matrix in the batch.
         *          The inverse might be calculated using the adjugate matrix (transpose of the cofactor matrix) and the determinant: `A^-1 = adj(A) / det(A)`.
         *          Handles non-invertible matrices (where determinant is close to zero) appropriately, possibly returning NaNs or infs in the inverse.
         * @param matrix Tensor of shape `[N, D, D]` representing a batch of square matrices.
         * @return A `std::tuple<torch::Tensor, torch::Tensor>` containing:
         *         - `[0]`: The inverse matrices, shape `[N, D, D]`.
         *         - `[1]`: The determinants, shape `[N]`.
         */
        std::tuple<torch::Tensor, torch::Tensor> Inverse(torch::Tensor matrix);

        /**
         * @brief Calculates the eigenvalues and eigenvectors for a batch of symmetric matrices.
         * @details Performs eigenvalue decomposition for each symmetric `[D, D]` matrix in the batch, finding `lambda` and `v` such that `A v = lambda v`.
         *          Requires the input matrices to be symmetric.
         * @param matrix Tensor of shape `[N, D, D]` representing a batch of symmetric square matrices.
         * @return A `std::tuple<torch::Tensor, torch::Tensor>` containing:
         *         - `[0]`: The eigenvalues, shape `[N, D]`. Usually sorted.
         *         - `[1]`: The eigenvectors, shape `[N, D, D]`. Each column `[:, :, i]` corresponds to the eigenvalue `[:, i]`. Eigenvectors form an orthonormal basis.
         */
        std::tuple<torch::Tensor, torch::Tensor> Eigenvalue(torch::Tensor matrix);

        /**
         * @brief Calculates the cross product of two batches of 3D vectors.
         * @details Computes `v = mat1 x mat2` for each pair of vectors in the batch.
         *          `v_x = m1_y*m2_z - m1_z*m2_y`
         *          `v_y = m1_z*m2_x - m1_x*m2_z`
         *          `v_z = m1_x*m2_y - m1_y*m2_x`
         *          Assumes input tensors represent 3D vectors.
         * @param mat1 Tensor of shape `[N, 3]` representing the first batch of vectors.
         * @param mat2 Tensor of shape `[N, 3]` representing the second batch of vectors.
         * @return Tensor of shape `[N, 3]` containing the cross products.
         */
        torch::Tensor Cross(torch::Tensor mat1, torch::Tensor mat2);
    } // namespace operators

    /**
     * @brief Namespace containing functions for neutrino momentum reconstruction, particularly relevant for ttbar events.
     * @details Provides implementations of analytical and numerical methods to solve for the unmeasured neutrino momenta
     *          based on kinematic constraints (e.g., W and top quark masses) and measured objects (leptons, b-jets, MET).
     *          Includes interfaces for both tensor-based inputs (suitable for GPU acceleration) and standard C++ vector/object inputs.
     */
    namespace nusol {
        /**
         * @brief Calculates base matrices and vectors used in the analytical single-neutrino reconstruction algorithm.
         * @details This function pre-computes several intermediate quantities derived from the input kinematics and mass constraints.
         *          These quantities (`A`, `B`, `C`, `a`, `b`, etc.) form the coefficients of the quadratic equation in the neutrino's Pz component,
         *          which arises from imposing the W boson and top quark mass constraints. See relevant physics papers for the derivation (e.g., formulas involving dot products and energies).
         * @param pmc_b Tensor of shape `[N, 4]` for b-jet four-momenta `(Px, Py, Pz, E)`.
         * @param pmc_mu Tensor of shape `[N, 4]` for lepton four-momenta `(Px, Py, Pz, E)`.
         * @param masses Tensor of shape `[N, 3]` containing mass constraints `[MassW, MassNu, MassTop]` for each event. Assumed consistent units (e.g., GeV).
         * @return A `torch::Dict<std::string, torch::Tensor>` containing intermediate matrices and vectors used in the `Nu` solver.
         *         Keys typically include `"A"`, `"B"`, `"C"` (coefficients of Pz^2, Pz, constant term) and potentially others like `"a"`, `"b"`.
         *         Tensor shapes depend on the specific quantity but are usually `[N]` or `[N, D, D]`.
         */
        torch::Dict<std::string, torch::Tensor> BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses);

        /**
         * @brief Performs analytical reconstruction of a single neutrino's four-momentum in semi-leptonic decays (e.g., t -> b W(-> l nu)).
         * @details Solves the system of equations derived from the W mass constraint `(p_l + p_nu)^2 = m_W^2` and the top mass constraint `(p_b + p_l + p_nu)^2 = m_top^2`.
         *          The transverse components of the neutrino momentum are assumed to be given by the missing transverse energy (MET): `p_nu_x = METx`, `p_nu_y = METy`.
         *          This leads to a quadratic equation for the longitudinal component `p_nu_z`. The function calculates the coefficients (potentially using `BaseMatrix`),
         *          solves the quadratic equation, yielding 0, 1, or 2 real solutions for `p_nu_z`. It may use the MET covariance matrix `sigma` to calculate a chi-squared
         *          value for each solution, ranking them or selecting the best one based on compatibility with MET resolution.
         * @param pmc_b Tensor of shape `[N, 4]` for the b-jet four-momentum `(Px, Py, Pz, E)`.
         * @param pmc_mu Tensor of shape `[N, 4]` for the lepton four-momentum `(Px, Py, Pz, E)`.
         * @param met_xy Tensor of shape `[N, 2]` for the missing transverse energy `(METx, METy)`.
         * @param masses Tensor of shape `[N, 3]` containing mass constraints `[MassW, MassNu, MassTop]` for each event.
         * @param sigma Tensor of shape `[N, 3, 3]` or `[N, 2, 2]` representing the covariance matrix for MET resolution (e.g., `[[cov(x,x), cov(x,y)], [cov(y,x), cov(y,y)]]`). Used for chi-squared calculation. Can be optional depending on implementation details.
         * @param null A small numerical tolerance value (default: 1e-10). Used for checking if the discriminant is close to zero (one solution) or if coefficients are negligible.
         * @return A `torch::Dict<std::string, torch::Tensor>` containing the reconstruction results:
         *         - `"Nu"`: Tensor of shape `[N, 2, 4]` containing the four-momenta `(Px, Py, Pz, E)` of the two possible neutrino solutions per event. Invalid solutions might be filled with NaNs or zeros. Px and Py are derived from `met_xy`.
         *         - `"delta"`: Tensor of shape `[N]` containing the discriminant of the quadratic equation (`b^2 - 4ac`). Indicates the nature of solutions (positive: 2 real, zero: 1 real, negative: complex).
         *         - `"chi2"`: Tensor of shape `[N, 2]` (optional) containing chi-squared values for each solution, calculated using `sigma`. Lower values indicate better agreement with MET resolution.
         *         - `"status"`: Tensor of shape `[N]` (optional) indicating the number of real solutions found (0, 1, or 2).
         */
        torch::Dict<std::string, torch::Tensor> Nu(
               torch::Tensor pmc_b , torch::Tensor pmc_mu, torch::Tensor met_xy,
               torch::Tensor masses, torch::Tensor  sigma, double null = 10e-10
        );

        /**
         * @brief Performs analytical reconstruction of two neutrino four-momenta in dileptonic events (e.g., ttbar -> b l nu b l nu).
         * @details This function attempts to solve the system of 8 unknowns (2x Px, Py, Pz, E for neutrinos) using constraints:
         *          - 2x W mass constraints: `(p_l1 + p_nu1)^2 = m_W1^2`, `(p_l2 + p_nu2)^2 = m_W2^2`
         *          - 2x Top mass constraints: `(p_b1 + p_l1 + p_nu1)^2 = m_top1^2`, `(p_b2 + p_l2 + p_nu2)^2 = m_top2^2`
         *          - 2x MET constraints: `p_nu1_x + p_nu2_x = METx`, `p_nu1_y + p_nu2_y = METy`
         *          This system is generally underconstrained. This implementation likely uses an iterative approach (e.g., based on Betchart et al., arXiv:1305.1878 or similar methods)
         *          that might scan possible energy fractions or use numerical minimization to find solutions compatible with the constraints.
         *          The parameters `step`, `tolerance`, and `timeout` control the iterative solver.
         * @param pmc_b1 Tensor of shape `[N, 4]` for the first b-jet `(Px, Py, Pz, E)`.
         * @param pmc_b2 Tensor of shape `[N, 4]` for the second b-jet `(Px, Py, Pz, E)`.
         * @param pmc_l1 Tensor of shape `[N, 4]` for the first lepton `(Px, Py, Pz, E)`.
         * @param pmc_l2 Tensor of shape `[N, 4]` for the second lepton `(Px, Py, Pz, E)`.
         * @param met_xy Tensor of shape `[N, 2]` for the missing transverse energy `(METx, METy)`.
         * @param masses Tensor of shape `[N, 5]` or `[N, 6]` containing mass constraints, typically `[MassW1, MassNu1, MassTop1, MassW2, MassNu2, MassTop2]`. Order might vary. Assumes consistent units.
         * @param null Small numerical tolerance value (default: 1e-10) for stability checks.
         * @param step Step size parameter for the iterative solver (default: 1e-9). Controls how much parameters are adjusted per iteration.
         * @param tolerance Convergence tolerance for the iterative solver (default: 1e-6). The solver stops if the change in solution or objective function falls below this threshold.
         * @param timeout Maximum number of iterations allowed for the solver (default: 1000). Prevents infinite loops.
         * @return A `torch::Dict<std::string, torch::Tensor>` containing the reconstruction results:
         *         - `"Nu1"`: Tensor of shape `[N, 4]` containing the reconstructed four-momentum `(Px, Py, Pz, E)` for the first neutrino.
         *         - `"Nu2"`: Tensor of shape `[N, 4]` containing the reconstructed four-momentum `(Px, Py, Pz, E)` for the second neutrino.
         *         - `"weight"`: Tensor of shape `[N]` (optional) representing a quality metric, likelihood, or weight associated with the found solution.
         *         - `"status"`: Tensor of shape `[N]` (optional) indicating the success or failure status of the solver for each event (e.g., converged, timed out, no solution found).
         */
        torch::Dict<std::string, torch::Tensor> NuNu(
               torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2,
               torch::Tensor met_xy, torch::Tensor masses, double null = 10e-10, const double step = 1e-9,
               const double tolerance = 1e-6, const unsigned int timeout = 1000
        );

        /**
         * @brief Overload for NuNu reconstruction where top mass constraints are provided separately, potentially assuming fixed W and neutrino masses.
         * @details Similar to the other `NuNu` function, but takes the two top mass constraints as separate tensors. This might be used in scenarios
         *          where `m_W` and `m_nu` are assumed fixed (e.g., PDG values), but the top masses might vary per event or per side (e.g., if testing different mass hypotheses).
         *          The underlying solver mechanism (iterative) and parameters (`null`, `step`, `tolerance`, `timeout`) are likely the same.
         *          Standard W and neutrino masses (e.g., 80.4 GeV, 0 GeV) are likely assumed internally.
         * @param pmc_b1 Tensor of shape `[N, 4]` for the first b-jet `(Px, Py, Pz, E)`.
         * @param pmc_b2 Tensor of shape `[N, 4]` for the second b-jet `(Px, Py, Pz, E)`.
         * @param pmc_l1 Tensor of shape `[N, 4]` for the first lepton `(Px, Py, Pz, E)`.
         * @param pmc_l2 Tensor of shape `[N, 4]` for the second lepton `(Px, Py, Pz, E)`.
         * @param met_xy Tensor of shape `[N, 2]` for the missing transverse energy `(METx, METy)`.
         * @param null Small numerical tolerance value for stability checks.
         * @param mass1 Tensor of shape `[N]` or `[N, 1]` for the mass constraint on the `(b1, l1, nu1)` system (Top 1 mass).
         * @param mass2 Tensor of shape `[N]` or `[N, 1]` for the mass constraint on the `(b2, l2, nu2)` system (Top 2 mass). Assumes consistent units with kinematics.
         * @param step Step size parameter for the iterative solver (default: 1e-9).
         * @param tolerance Convergence tolerance for the iterative solver (default: 1e-6).
         * @param timeout Maximum number of iterations allowed for the solver (default: 1000).
         * @return A `torch::Dict<std::string, torch::Tensor>` containing the reconstructed neutrino solutions, with a structure similar to the other `NuNu` function
         *         (e.g., keys `"Nu1"`, `"Nu2"`, `"weight"`, `"status"`).
         */
        torch::Dict<std::string, torch::Tensor> NuNu(
               torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2,
               torch::Tensor met_xy, double null, torch::Tensor mass1, torch::Tensor mass2, const double step = 1e-9,
               const double tolerance = 1e-6, const unsigned int timeout = 1000
        );

        /**
         * @brief C++ interface for NuNu reconstruction using `std::vector` inputs for kinematics and masses.
         * @details This function serves as a bridge between standard C++ data structures and the tensor-based `NuNu` implementation.
         *          It takes pointers to vectors of kinematics (`std::vector<std::vector<double>>`), MET (`std::vector<double>`),
         *          MET phi (`std::vector<double>`), and mass constraints (`std::vector<std::vector<double>>`).
         *          It converts these inputs into `torch::Tensor` objects (likely placing them on the specified `dev` device),
         *          calculates METx/METy from MET magnitude and phi, calls the appropriate tensor-based `NuNu` function,
         *          and converts the resulting neutrino four-momenta back into `neutrino` objects.
         * @param pmc_b1 Pointer to `vector<vector<double>>` for first b-jet kinematics `[[px, py, pz, e], ...]`. Size `N`.
         * @param pmc_b2 Pointer to `vector<vector<double>>` for second b-jet kinematics. Size `N`.
         * @param pmc_l1 Pointer to `vector<vector<double>>` for first lepton kinematics. Size `N`.
         * @param pmc_l2 Pointer to `vector<vector<double>>` for second lepton kinematics. Size `N`.
         * @param met Pointer to `vector<double>` for MET magnitudes per event. Size `N`.
         * @param phi Pointer to `vector<double>` for MET phi angles (in radians) per event. Size `N`.
         * @param mass1 Pointer to `vector<vector<double>>` for mass constraints on side 1 `[[mW1, mNu1, mTop1], ...]`. Size `N`.
         * @param mass2 Pointer to `vector<vector<double>>` for mass constraints on side 2 `[[mW2, mNu2, mTop2], ...]`. Size `N`.
         * @param dev Device string specifying the computation device (e.g., `"cpu"`, `"cuda:0"`).
         * @param null Small numerical tolerance value passed to the solver.
         * @param step Step size parameter passed to the iterative solver.
         * @param tolerance Convergence tolerance passed to the iterative solver.
         * @param timeout Maximum number of iterations passed to the iterative solver.
         * @return A `std::vector` of `std::pair<neutrino*, neutrino*>`. Each pair corresponds to an input event and contains pointers
         *         to the two reconstructed `neutrino` objects. The `neutrino` objects contain the calculated four-momenta.
         *         The caller takes ownership of the `neutrino` objects pointed to in the pairs and is responsible for deleting them.
         *         The size of the vector is `N`. Indices `l_idx` and `b_idx` in the returned neutrinos likely correspond to the event index `0..N-1`.
         */
        std::vector<std::pair<neutrino*, neutrino*>> NuNu(
               std::vector<std::vector<double>>* pmc_b1, std::vector<std::vector<double>>* pmc_b2,
               std::vector<std::vector<double>>* pmc_l1, std::vector<std::vector<double>>* pmc_l2,
                           std::vector<double>*    met,             std::vector<double>*  phi,
               std::vector<std::vector<double>>* mass1, std::vector<std::vector<double>>* mass2,
               std::string dev, const double null, const double step, const double tolerance, const unsigned int timeout
        );

        /**
         * @brief Template C++ interface for NuNu reconstruction using vectors of generic particle type pointers.
         * @details This high-level interface allows using custom particle classes (like `particle_template` or derived classes) directly.
         *          It first extracts the Cartesian four-momenta from the input particle vectors using `pyc::to_pmc`.
         *          Then, it calls the `std::vector`-based `NuNu` function (which handles tensor conversion and computation).
         *          Finally, it populates the `bquark` and `lepton` pointers within the resulting `neutrino` objects.
         *          It creates *new copies* of the original input particle objects and assigns pointers to these copies
         *          to the `bquark` and `lepton` members of the corresponding `neutrino` objects.
         * @tparam b The type of the b-quark objects. Must be convertible to `std::vector<double>{px, py, pz, e}` by `pyc::as_pmc` and constructible via copy constructor or similar mechanism used by `new particle_template(...)`.
         * @tparam l The type of the lepton objects. Must be convertible to `std::vector<double>{px, py, pz, e}` by `pyc::as_pmc` and constructible via copy constructor or similar mechanism used by `new particle_template(...)`.
         * @param bquark1 Vector of pointers to the first b-quark objects. Size `N`.
         * @param bquark2 Vector of pointers to the second b-quark objects. Size `N`.
         * @param lepton1 Vector of pointers to the first lepton objects. Size `N`.
         * @param lepton2 Vector of pointers to the second lepton objects. Size `N`.
         * @param met_ Vector of MET magnitudes per event. Size `N`.
         * @param phi_ Vector of MET phi angles (in radians) per event. Size `N`.
         * @param mass1 Vector of `vector<double>` for mass constraints on side 1 `[[mW1, mNu1, mTop1], ...]`. Size `N`.
         * @param mass2 Vector of `vector<double>` for mass constraints on side 2 `[[mW2, mNu2, mTop2], ...]`. Size `N`.
         * @param dev Device string specifying the computation device (e.g., `"cpu"`, `"cuda:0"`).
         * @param null Small numerical tolerance value passed to the solver.
         * @param step Step size parameter passed to the iterative solver.
         * @param tolerance Convergence tolerance passed to the iterative solver.
         * @param timeout Maximum number of iterations passed to the iterative solver.
         * @return A `std::vector` of `std::pair<neutrino*, neutrino*>`. Each pair corresponds to an input event.
         *         The `neutrino` objects contain the reconstructed four-momenta and pointers (`bquark`, `lepton`)
         *         to *newly allocated copies* of the original input particles associated with that neutrino solution.
         *         The `l_idx` and `b_idx` members of the neutrinos store the original index of the lepton/b-quark in the input vectors.
         *         The caller takes ownership of the `neutrino` objects *and* the copied `particle_template` objects pointed to by their `bquark` and `lepton` members.
         *         The size of the vector is `N`.
         */
        template <typename b, typename l>
        std::vector<std::pair<neutrino*, neutrino*>> NuNu(
               std::vector<b*> bquark1, std::vector<b*> bquark2,
               std::vector<l*> lepton1, std::vector<l*> lepton2,
               std::vector<double> met_, std::vector<double> phi_,
               std::vector<std::vector<double>> mass1, std::vector<std::vector<double>> mass2,
               std::string dev, double null, const double step, const double tolerance, const unsigned int timeout
        );

        /**
         * @brief Performs combinatorial neutrino reconstruction, likely for dileptonic ttbar events, testing different particle assignments.
         * @details In events with multiple b-jet and lepton candidates, this function likely explores different pairings
         *          (e.g., which b-jet goes with which lepton) to form the two `t -> b l nu` decay chains.
         *          For each combination, it might use an analytical solver (like `Nu` or `NuNu` variants) or another method
         *          (e.g., kinematic fit, likelihood minimization) to reconstruct the neutrinos and evaluate the quality of the combination
         *          based on the W and top mass constraints (`mW`, `mT`). It selects the best combination per event based on some metric (e.g., lowest chi2, highest likelihood).
         *          The `edge_index`, `batch`, `pmc`, `pid` inputs suggest a graph-based representation of events, where nodes are particles and edges might represent potential pairings or proximity.
         * @param edge_index Tensor of shape `[2, num_edges]` defining graph connectivity (e.g., pairs of particles to consider). May not be strictly required depending on the combinatorial method.
         * @param batch Tensor of shape `[num_nodes]` assigning each particle (node) in `pmc` to an event index. Used to group particles by event.
         * @param pmc Tensor of particle four-momenta, shape `[num_nodes, 4]` `(Px, Py, Pz, E)`. Contains all particles across all events.
         * @param pid Tensor of particle type identifiers (e.g., PDG ID), shape `[num_nodes]`. Used to identify b-jets (e.g., `abs(pid)==5`) and leptons (e.g., `abs(pid)==11` or `13`).
         * @param met_xy Tensor of MET vectors, shape `[num_events, 2]` `(METx, METy)`. The number of events should correspond to the unique values in `batch`.
         * @param mT Target top quark mass constraint (default: 172.62 GeV = 172620 MeV). Units should match `pmc`.
         * @param mW Target W boson mass constraint (default: 80.385 GeV = 80385 MeV). Units should match `pmc`.
         * @param null Small numerical tolerance value for internal solvers (default: 1e-10).
         * @param perturb Small perturbation factor, possibly used in numerical optimization or stability checks (default: 1e-3).
         * @param steps Number of steps or iterations if a numerical optimization method is used (default: 100).
         * @param gev If `true`, assume input `mT` and `mW` are in GeV and convert them internally to MeV (by multiplying by 1000). If `false` (default), assume `mT` and `mW` are already in units consistent with `pmc` (typically MeV).
         * @return A `torch::Dict<std::string, torch::Tensor>` containing results. Structure depends on the method, but might include:
         *         - `"Nu1"`, `"Nu2"`: Reconstructed neutrino four-momenta for the best combination per event, shape `[num_events, 4]`.
         *         - `"Indices"`: Information about which original particles formed the best combination (e.g., indices of the chosen b-jets and leptons per event).
         *         - `"Score"`: The quality score (e.g., chi2, likelihood) of the best combination per event.
         *         - `"Status"`: Success/failure status per event.
         */
        torch::Dict<std::string, torch::Tensor> combinatorial(
               torch::Tensor edge_index, torch::Tensor batch , torch::Tensor pmc, torch::Tensor pid, torch::Tensor met_xy,
               double mT  = 172.62*1000 , double mW = 80.385*1000, double null = 1e-10, double perturb = 1e-3,
               long steps = 100, bool gev = false
        );

        /**
         * @brief C++ interface for combinatorial reconstruction using `std::vector` inputs.
         * @details This function acts as a wrapper around the tensor-based `combinatorial` function.
         *          It takes particle kinematics (`pmc`), event assignments (`bth`), particle type flags (`is_b`, `is_l`),
         *          and MET information (`met_`, `phi_`) as standard C++ vectors.
         *          It converts these inputs into the required `torch::Tensor` format (including calculating `met_xy` and potentially building `edge_index` if needed),
         *          calls the tensor-based implementation on the specified device `dev`, and converts the results (best-fit neutrino pair per event)
         *          back into a vector of `neutrino` object pairs.
         * @param met_ Pointer to `vector<double>` for MET magnitudes per event. Size `num_events`.
         * @param phi_ Pointer to `vector<double>` for MET phi angles (in radians) per event. Size `num_events`.
         * @param pmc Pointer to `vector<vector<double>>` containing all particle kinematics `[[px, py, pz, e], ...]` concatenated across events. Size `num_nodes`.
         * @param bth Pointer to `vector<long>` indicating the batch index (event number, 0 to `num_events-1`) for each particle in `pmc`. Size `num_nodes`.
         * @param is_b Pointer to `vector<long>` flag (0 or 1) indicating if particle `i` is a b-jet candidate. Size `num_nodes`.
         * @param is_l Pointer to `vector<long>` flag (0 or 1) indicating if particle `i` is a lepton candidate. Size `num_nodes`.
         * @param dev Device string specifying the computation device (e.g., `"cpu"`, `"cuda:0"`).
         * @param mT Target top quark mass constraint (in MeV, or GeV if internal conversion happens based on a potential `gev` flag not shown here).
         * @param mW Target W boson mass constraint (in MeV, or GeV).
         * @param null Small numerical tolerance value passed to the solver.
         * @param perturb Perturbation factor passed to the solver.
         * @param steps Number of steps/iterations passed to the solver.
         * @return A `std::vector` of `std::pair<neutrino*, neutrino*>`, size `num_events`. Each pair represents the best-fit reconstructed
         *         neutrino pair for the corresponding event. The `neutrino` objects contain four-momenta and potentially indices (`l_idx`, `b_idx`)
         *         referring back to the original particle indices within their event that formed the best combination.
         *         The caller takes ownership of the `neutrino` objects.
         */
        std::vector<std::pair<neutrino*, neutrino*>> combinatorial(
                std::vector<double>* met_, std::vector<double>* phi_, std::vector<std::vector<double>>* pmc,
                std::vector<long>* bth, std::vector<long>* is_b, std::vector<long>* is_l, std::string dev,
                double mT, double mW, double null, double perturb, long steps
        );

        /**
         * @brief C++ interface for combinatorial reconstruction using vectors of `particle_template` pointers, grouped by event.
         * @details This high-level interface takes input particles organized per event.
         *          It likely first flattens the `particles` structure into concatenated vectors for kinematics (`pmc`),
         *          batch indices (`bth`), and type flags (`is_b`, `is_l`) based on information within `particle_template` (assuming it stores type).
         *          It then calls the `std::vector`-based `combinatorial` function (which handles tensor conversion and computation).
         *          Finally, it returns the best-fit neutrino pair for each event. It might populate the `bquark` and `lepton` pointers
         *          in the returned `neutrino` objects, potentially pointing to copies of the original particles involved in the best combination.
         * @param met_ Vector of MET magnitudes per event. Size `num_events`.
         * @param phi_ Vector of MET phi angles (in radians) per event. Size `num_events`.
         * @param particles Vector where each element `particles[i]` is a `std::vector<particle_template*>` containing all candidate particles for event `i`. Size `num_events`. The `particle_template` objects should contain kinematics and allow identification of b-jets and leptons.
         * @param dev Device string specifying the computation device (e.g., `"cpu"`, `"cuda:0"`).
         * @param mT Target top quark mass constraint (in MeV, or GeV).
         * @param mW Target W boson mass constraint (in MeV, or GeV).
         * @param null Small numerical tolerance value passed to the solver.
         * @param perturb Perturbation factor passed to the solver.
         * @param steps Number of steps/iterations passed to the solver.
         * @return A `std::vector` of `std::pair<neutrino*, neutrino*>`, size `num_events`. Each pair represents the best-fit reconstructed
         *         neutrino pair for the corresponding event. The `neutrino` objects contain four-momenta. They might also contain populated `bquark` and `lepton` pointers (potentially to copies) and indices (`l_idx`, `b_idx`) indicating the original particles in the best combination.
         *         The caller takes ownership of the `neutrino` objects and any associated copied particles.
         */
        std::vector<std::pair<neutrino*, neutrino*>> combinatorial(
               std::vector<double> met_, std::vector<double> phi_,
               std::vector<std::vector<particle_template*>> particles,
               std::string dev, double mT, double mW, double null, double perturb, long steps
        );
    } // namespace nusol

    /**
     * @brief Namespace containing functions related to graph operations, often used in the context of Graph Neural Networks (GNNs) or graph-based analysis.
     * @details Provides tools for aggregating information (node features, edge predictions) across graph structures defined by edge indices.
     *          Includes functions for common graph algorithms like PageRank and specialized aggregations for physics analysis (e.g., summing kinematics).
     */
    namespace graph {
        /**
         * @brief Aggregates node features based on graph edges and edge predictions/weights, typically summing weighted features of neighbors.
         * @details Implements a message passing step common in GNNs or graph analysis. For each edge `(j -> i)` defined in `edge_index` (where `j` is source, `i` is destination),
         *          it takes the feature vector of the source node `j` (`node_feature[j]`), scales it by the corresponding edge `prediction` (which could be a scalar weight or a vector),
         *          and aggregates (e.g., sums) these weighted features at the destination node `i`. The exact direction (source-to-destination or vice-versa) depends on convention.
         * @param edge_index Tensor of shape `[2, num_edges]` defining graph connectivity. `edge_index[0]` = source nodes, `edge_index[1]` = destination nodes. Assumed 0-indexed.
         * @param prediction Tensor of shape `[num_edges]` (scalar weight per edge) or `[num_edges, num_classes]` (vector weight/score per edge). Used to scale node features during aggregation.
         * @param node_feature Tensor of shape `[num_nodes, feature_dim]` containing features associated with each node.
         * @return A `torch::Dict<std::string, torch::Tensor>` containing the aggregated features.
         *         A common output key is `"edge_agg"` (or similar), holding a tensor of shape `[num_nodes, feature_dim]` (if prediction is scalar) or `[num_nodes, feature_dim * num_classes]` (or similar, depending on how vector predictions are handled), representing the aggregated weighted features arriving at each node.
         */
        torch::Dict<std::string, torch::Tensor> edge_aggregation(
            torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature
        );

        /**
         * @brief Aggregates edge predictions/weights onto nodes.
         * @details For each node `i`, this function aggregates the `prediction` values associated with edges connected to it.
         *          The aggregation could be based on incoming edges (summing `prediction` for all edges `(j -> i)`) or outgoing edges (summing for `(i -> j)`),
         *          or both, depending on the implementation. Common aggregation methods include sum, mean, max.
         * @param edge_index Tensor of shape `[2, num_edges]` defining graph connectivity.
         * @param prediction Tensor of shape `[num_edges]` or `[num_edges, prediction_dim]` containing edge weights or scores to be aggregated.
         * @param node_feature Tensor of shape `[num_nodes, feature_dim]`. Potentially unused in this function but kept for signature consistency with `edge_aggregation`.
         * @return A `torch::Dict<std::string, torch::Tensor>` containing the aggregated edge information per node.
         *         A common output key is `"node_agg"` (or similar), holding a tensor of shape `[num_nodes, prediction_dim]` (if prediction has dimension `prediction_dim`), representing the aggregated edge scores/weights associated with each node.
         */
        torch::Dict<std::string, torch::Tensor> node_aggregation(
            torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature
        );

        /**
         * @brief Aggregates features based on a cluster mapping, grouping items belonging to the same cluster.
         * @details Takes a tensor of features and a mapping that assigns each feature vector (row) to a cluster index.
         *          It groups the feature vectors by their assigned cluster index and performs an aggregation (e.g., sum, mean)
         *          within each cluster. This is useful for pooling node features into cluster-level representations.
         * @param cluster_map Tensor of shape `[num_items]` where `cluster_map[k]` is the integer index of the cluster that item `k` belongs to. Assumed 0-indexed and contiguous cluster indices.
         * @param features Tensor of shape `[num_items, feature_dim]` containing the features to be aggregated.
         * @return A `torch::Dict<std::string, torch::Tensor>` containing the aggregated features per cluster.
         *         A common output key is `"cluster_agg"` (or similar), holding a tensor of shape `[num_clusters, feature_dim]`, where `num_clusters` is the number of unique indices in `cluster_map`.
         *         The dictionary might also include tensors mapping original items to their aggregated cluster features.
         */
        torch::Dict<std::string, torch::Tensor> unique_aggregation(
                torch::Tensor cluster_map, torch::Tensor features
        );

        /**
         * @brief Computes PageRank scores for nodes in a graph, potentially using edge scores as weights.
         * @details Implements the PageRank algorithm iteratively. The rank of a node is determined by the ranks of nodes pointing to it, weighted by the edge scores (if provided)
         *          and normalized by the total outgoing weight of the source nodes. The `alpha` parameter is the damping factor (probability of following an edge vs. teleporting).
         *          The iteration continues until the ranks converge (change less than `threshold`) or the `timeout` is reached.
         *          Can compute PageRank for multiple classes simultaneously if `edge_scores` has multiple columns (`num_cls > 1`).
         * @param edge_index Tensor of shape `[2, num_edges]` defining graph connectivity (source, destination).
         * @param edge_scores Tensor of shape `[num_edges]` or `[num_edges, num_cls]` representing the weight/importance of each edge. Higher scores mean stronger links.
         * @param alpha Damping factor for PageRank (probability of random jump). Typically 0.85. Value should be in `[0, 1]`. (Default: 0.85)
         * @param threshold Convergence threshold for the iterative PageRank calculation. Iteration stops when the maximum change in any node's rank is below this value. (Default: 0.5 - Note: This seems unusually high, might represent relative change or a different metric. Typical values are much smaller, e.g., 1e-6).
         * @param norm_low Minimum value added to normalization denominators (sum of outgoing edge weights) to prevent division by zero or instability. (Default: 1e-6).
         * @param timeout Maximum number of iterations for PageRank calculation. (Default: 1,000,000).
         * @param num_cls Number of classes or components for which to calculate PageRank independently. Should match the second dimension of `edge_scores` if it's 2D. (Default: 2).
         * @return A `torch::Dict<std::string, torch::Tensor>` containing the PageRank scores.
         *         Key `"PageRank"` likely holds a tensor of shape `[num_nodes]` (if `num_cls=1`) or `[num_nodes, num_cls]` containing the final PageRank score for each node (and potentially each class). Scores are typically non-negative and might sum to 1 over all nodes (per class).
         */
        torch::Dict<std::string, torch::Tensor> PageRank(
                torch::Tensor edge_index, torch::Tensor edge_scores,
                double alpha = 0.85, double threshold = 0.5, double norm_low = 1e-6, long timeout = 1e6, long num_cls = 2
        );

        /**
         * @brief Computes PageRank scores and uses them to perform a weighted reconstruction (sum) of particle kinematics.
         * @details This function first calculates PageRank scores for each node (and potentially each class) using the `PageRank` function with the provided graph structure (`edge_index`),
         *          edge weights (`edge_scores`), and PageRank parameters (`alpha`, `threshold`, etc.).
         *          It then uses these PageRank scores as weights to compute a weighted sum of the particle four-momenta (`pmc`).
         *          The summation is likely performed per class, resulting in a reconstructed four-momentum for each class. This could be used, for example,
         *          to reconstruct the four-momentum of a decaying particle by summing the PageRank-weighted momenta of its decay products identified by the graph edges and scores.
         * @param edge_index Tensor of shape `[2, num_edges]` defining graph connectivity.
         * @param edge_scores Tensor of shape `[num_edges]` or `[num_edges, num_cls]` representing edge weights/scores used for PageRank calculation.
         * @param pmc Tensor of shape `[num_nodes, 4]` containing the Cartesian four-momenta `(Px, Py, Pz, E)` of the nodes (particles).
         * @param alpha Damping factor for PageRank. (Default: 0.85).
         * @param threshold Convergence threshold for PageRank. (Default: 0.5 - see note in `PageRank`).
         * @param norm_low Minimum value for normalization denominators in PageRank. (Default: 1e-6).
         * @param timeout Maximum number of iterations for PageRank. (Default: 1,000,000).
         * @param num_cls Number of classes/components for PageRank and reconstruction. (Default: 2). Determines the number of reconstructed momenta.
         * @return A `torch::Dict<std::string, torch::Tensor>` containing the results:
         *         - `"PageRank"`: Tensor of shape `[num_nodes, num_cls]` containing the calculated PageRank scores.
         *         - `"Reconstruction"`: Tensor of shape `[num_cls, 4]` containing the reconstructed four-momenta, obtained by summing `pmc` weighted by `PageRank` for each class.
         *         The dictionary might contain results per event if the input graph represents multiple events (requires a `batch` tensor, which is missing in the signature but might be implicitly handled). If operating on a single graph, the output shape is as described. If operating on batches, the Reconstruction shape might be `[num_events, num_cls, 4]`.
         */
        torch::Dict<std::string, torch::Tensor> PageRankReconstruction(
                torch::Tensor edge_index, torch::Tensor edge_scores, torch::Tensor pmc,
                double alpha = 0.85, double threshold = 0.5, double norm_low = 1e-6, long timeout = 1e6, long num_cls = 2
        );

        /**
         * @brief Namespace for graph aggregations specifically using Polar coordinates (Pt, Eta, Phi, E/M).
         * @details Provides versions of `edge_aggregation` and `node_aggregation` tailored for inputs in polar coordinates.
         *          Note that aggregating polar coordinates directly (especially angles like Phi) requires careful handling (e.g., converting to Cartesian, summing, converting back).
         *          The implementation details determine how this is handled.
         */
        namespace polar {
            /**
             * @brief Aggregates polar kinematics based on edges and predictions.
             * @details Similar to the Cartesian `edge_aggregation`, but takes node features as polar kinematics `pmu = (Pt, Eta, Phi, E or M)`.
             *          The aggregation likely involves converting polar to Cartesian coordinates, performing a weighted sum using `prediction`,
             *          and potentially converting the result back to polar coordinates, or returning the summed Cartesian vector.
             * @param edge_index Tensor of shape `[2, num_edges]` defining graph connectivity.
             * @param prediction Tensor of shape `[num_edges]` or `[num_edges, num_classes]` containing edge weights/scores.
             * @param pmu Tensor of shape `[num_nodes, 4]` containing polar kinematics `(Pt, Eta, Phi, E or M)`.
             * @return A `torch::Dict<std::string, torch::Tensor>` containing aggregated kinematics.
             *         The output format (polar or Cartesian) and shape (e.g., `[num_nodes, 4]`) depend on the implementation. Key might be `"edge_agg"`.
             */
            torch::Dict<std::string, torch::Tensor> edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
            );

            /**
             * @brief Aggregates edge predictions onto nodes, using polar kinematics as context (potentially unused).
             * @details Similar to the Cartesian `node_aggregation`, aggregating `prediction` values onto nodes based on `edge_index`.
             *          The `pmu` tensor might be unused but is kept for signature consistency.
             * @param edge_index Tensor of shape `[2, num_edges]` defining graph connectivity.
             * @param prediction Tensor of shape `[num_edges]` or `[num_edges, prediction_dim]` containing edge weights/scores.
             * @param pmu Tensor of shape `[num_nodes, 4]` containing polar kinematics (potentially unused).
             * @return A `torch::Dict<std::string, torch::Tensor>` containing aggregated edge predictions per node.
             *         Output shape e.g., `[num_nodes, prediction_dim]`. Key might be `"node_agg"`.
             */
            torch::Dict<std::string, torch::Tensor> node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
            );

            /**
             * @brief Aggregates polar kinematics (provided separately) based on edges and predictions.
             * @details Version of `edge_aggregation` where polar components (Pt, Eta, Phi, E) are provided as separate tensors.
             *          Likely converts these to Cartesian coordinates internally before performing the weighted aggregation.
             * @param edge_index Tensor of shape `[2, num_edges]`.
             * @param prediction Tensor of shape `[num_edges]` or `[num_edges, num_classes]`.
             * @param pt Tensor of shape `[num_nodes]` containing transverse momentum.
             * @param eta Tensor of shape `[num_nodes]` containing pseudorapidity.
             * @param phi Tensor of shape `[num_nodes]` containing azimuthal angle (radians).
             * @param e Tensor of shape `[num_nodes]` containing energy.
             * @return A `torch::Dict<std::string, torch::Tensor>` containing aggregated kinematics (likely Cartesian sum), shape e.g., `[num_nodes, 4]`.
             */
            torch::Dict<std::string, torch::Tensor> edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction,
                torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
            );

            /**
             * @brief Aggregates edge predictions onto nodes, with separate polar kinematics as context (potentially unused).
             * @details Version of `node_aggregation` where polar components are provided separately. These components are likely unused.
             * @param edge_index Tensor of shape `[2, num_edges]`.
             * @param prediction Tensor of shape `[num_edges]` or `[num_edges, prediction_dim]`.
             * @param pt Tensor of shape `[num_nodes]` (potentially unused).
             * @param eta Tensor of shape `[num_nodes]` (potentially unused).
             * @param phi Tensor of shape `[num_nodes]` (potentially unused).
             * @param e Tensor of shape `[num_nodes]` (potentially unused).
             * @return A `torch::Dict<std::string, torch::Tensor>` containing aggregated edge predictions per node, shape e.g., `[num_nodes, prediction_dim]`.
             */
            torch::Dict<std::string, torch::Tensor> node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction,
                torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
            );
        } // namespace polar

        /**
         * @brief Namespace for graph aggregations specifically using Cartesian coordinates (Px, Py, Pz, E).
         * @details Provides versions of `edge_aggregation` and `node_aggregation` that explicitly work with Cartesian four-momenta.
         *          Aggregation typically involves straightforward weighted summation of four-vectors.
         */
        namespace cartesian {
            /**
             * @brief Aggregates Cartesian kinematics based on edges and predictions by summing weighted four-vectors.
             * @details For each edge `(j -> i)`, scales the four-vector `pmc[j]` by `prediction` and adds it to the aggregation for node `i`.
             *          This directly sums the four-momenta according to the graph structure and edge weights.
             * @param edge_index Tensor of shape `[2, num_edges]` defining graph connectivity.
             * @param prediction Tensor of shape `[num_edges]` or `[num_edges, num_classes]` containing edge weights/scores.
             * @param pmc Tensor of shape `[num_nodes, 4]` containing Cartesian kinematics `(Px, Py, Pz, E)`.
             * @return A `torch::Dict<std::string, torch::Tensor>` containing aggregated Cartesian kinematics.
             *         Output shape e.g., `[num_nodes, 4]` (if prediction is scalar) or `[num_nodes, 4 * num_classes]` (or similar). Key might be `"edge_agg"`.
             */
            torch::Dict<std::string, torch::Tensor> edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
            );

            /**
             * @brief Aggregates edge predictions onto nodes, using Cartesian kinematics as context (potentially unused).
             * @details Similar to the base `node_aggregation`, aggregating `prediction` values onto nodes based on `edge_index`.
             *          The `pmc` tensor might be unused but is kept for signature consistency.
             * @param edge_index Tensor of shape `[2, num_edges]` defining graph connectivity.
             * @param prediction Tensor of shape `[num_edges]` or `[num_edges, prediction_dim]` containing edge weights/scores.
             * @param pmc Tensor of shape `[num_nodes, 4]` containing Cartesian kinematics (potentially unused).
             * @return A `torch::Dict<std::string, torch::Tensor>` containing aggregated edge predictions per node.
             *         Output shape e.g., `[num_nodes, prediction_dim]`. Key might be `"node_agg"`.
             */
            torch::Dict<std::string, torch::Tensor> node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
            );

            /**
             * @brief Aggregates Cartesian kinematics (provided separately) based on edges and predictions.
             * @details Version of `edge_aggregation` where Cartesian components (Px, Py, Pz, E) are provided as separate tensors.
             *          Stacks them into a temporary `[N, 4]` tensor internally before performing the weighted aggregation.
             * @param edge_index Tensor of shape `[2, num_edges]`.
             * @param prediction Tensor of shape `[num_edges]` or `[num_edges, num_classes]`.
             * @param px Tensor of shape `[num_nodes]` containing x-momentum.
             * @param py Tensor of shape `[num_nodes]` containing y-momentum.
             * @param pz Tensor of shape `[num_nodes]` containing z-momentum.
             * @param e Tensor of shape `[num_nodes]` containing energy.
             * @return A `torch::Dict<std::string, torch::Tensor>` containing aggregated Cartesian kinematics, shape e.g., `[num_nodes, 4]`.
             */
            torch::Dict<std::string, torch::Tensor> edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction,
                torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
            );

            /**
             * @brief Aggregates edge predictions onto nodes, with separate Cartesian kinematics as context (potentially unused).
             * @details Version of `node_aggregation` where Cartesian components are provided separately. These components are likely unused.
             * @param edge_index Tensor of shape `[2, num_edges]`.
             * @param prediction Tensor of shape `[num_edges]` or `[num_edges, prediction_dim]`.
             * @param px Tensor of shape `[num_nodes]` (potentially unused).
             * @param py Tensor of shape `[num_nodes]` (potentially unused).
             * @param pz Tensor of shape `[num_nodes]` (potentially unused).
             * @param e Tensor of shape `[num_nodes]` (potentially unused).
             * @return A `torch::Dict<std::string, torch::Tensor>` containing aggregated edge predictions per node, shape e.g., `[num_nodes, prediction_dim]`.
             */
            torch::Dict<std::string, torch::Tensor> node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction,
                torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
            );
        } // namespace cartesian
    } // namespace graph
} // namespace pyc

#endif // PYC_H
