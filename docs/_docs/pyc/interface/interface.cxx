/**
 * @brief Destructor for the `neutrino` class.
 * @details This destructor is responsible for cleaning up any resources held by a `neutrino` object.
 *          Specifically, it handles the deallocation of dynamically allocated memory associated
 *          with the b-quark and lepton pointers that might be members of the `neutrino` class,
 *          preventing memory leaks when a `neutrino` object goes out of scope or is explicitly deleted.
 */
neutrino::~neutrino();

/**
 * @brief Converts a pointer to a `std::map` of string keys and `torch::Tensor` values into a `torch::Dict`.
 * @details This function facilitates the conversion between standard C++ map containers and PyTorch's dictionary
 *          type (`torch::Dict`). It takes a pointer to the input map to avoid copying the entire map if it's large.
 *          The resulting `torch::Dict` will have the same key-value pairs as the input map.
 * @param inpt A pointer to the `std::map<std::string, torch::Tensor>` that needs to be converted.
 *             The map should contain string keys and `torch::Tensor` values.
 * @return A `torch::Dict<std::string, torch::Tensor>` object containing the data from the input map.
 *         If the input pointer is null or points to an empty map, an empty `torch::Dict` might be returned.
 */
torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor>* inpt);

/**
 * @brief Converts a `std::map` of string keys and `torch::Tensor` values into a `torch::Dict`.
 * @details This function serves the same purpose as the pointer version but operates on a map passed by value.
 *          This implies that a copy of the input map might be made. It converts the standard C++ map
 *          into PyTorch's `torch::Dict` format, preserving the key-value pairs.
 * @param inpt The `std::map<std::string, torch::Tensor>` to be converted. The map is passed by value.
 * @return A `torch::Dict<std::string, torch::Tensor>` object representing the converted map.
 */
torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor> inpt);

/**
 * @brief Converts a pointer to a `std::vector` of doubles into a `torch::Tensor`.
 * @details This utility function transforms a standard C++ vector containing double-precision floating-point
 *          numbers into a PyTorch tensor. It takes a pointer to the vector to potentially avoid copying
 *          large amounts of data. The resulting tensor will have a data type compatible with doubles (e.g., `torch::kFloat64`)
 *          and will contain the elements of the input vector.
 * @param inpt A pointer to the `std::vector<double>` to be converted.
 * @return A `torch::Tensor` containing the elements from the input vector. The tensor will have a shape
 *         corresponding to the size of the vector. If the input pointer is null or points to an empty vector,
 *         an empty tensor might be returned.
 */
torch::Tensor pyc::tensorize(std::vector<double>* inpt);

/**
 * @brief Converts a pointer to a `std::vector` of longs into a `torch::Tensor`.
 * @details Similar to the double version, this function converts a standard C++ vector of long integers
 *          into a PyTorch tensor. It accepts a pointer to the input vector. The resulting tensor will
 *          have a data type compatible with longs (e.g., `torch::kInt64`) and will hold the vector's elements.
 * @param inpt A pointer to the `std::vector<long>` to be converted.
 * @return A `torch::Tensor` containing the elements from the input vector. The tensor's shape will match
 *         the vector's size. If the input pointer is null or points to an empty vector, an empty tensor might be returned.
 */
torch::Tensor pyc::tensorize(std::vector<long>* inpt);

/**
 * @brief Converts a pointer to a `std::vector` of `std::vector` of doubles into a `torch::Tensor`.
 * @details This function handles the conversion of a 2D structure, represented as a vector of vectors
 *          containing doubles, into a 2D PyTorch tensor. It takes a pointer to the outer vector.
 *          The resulting tensor will typically have a shape of [outer_vector_size, inner_vector_size]
 *          (assuming inner vectors have consistent sizes) and a data type like `torch::kFloat64`.
 * @param inpt A pointer to the `std::vector<std::vector<double>>` to be converted.
 * @return A `torch::Tensor` representing the 2D data structure. If the input pointer is null, points to an
 *         empty outer vector, or contains empty inner vectors, the resulting tensor's shape and content
 *         will reflect this (potentially an empty tensor or a tensor with a dimension of size zero).
 */
torch::Tensor pyc::tensorize(std::vector<std::vector<double>>* inpt);

/**
 * @brief Converts a pointer to a `std::vector` of `std::vector` of longs into a `torch::Tensor`.
 * @details Analogous to the double version, this function converts a 2D structure represented as a
 *          vector of vectors of long integers into a 2D PyTorch tensor. It accepts a pointer to the
 *          outer vector. The resulting tensor will have a shape reflecting the input structure and a
 *          data type like `torch::kInt64`.
 * @param inpt A pointer to the `std::vector<std::vector<long>>` to be converted.
 * @return A `torch::Tensor` representing the 2D data structure. Similar considerations regarding null pointers,
 *         empty vectors apply as in the double version.
 */
torch::Tensor pyc::tensorize(std::vector<std::vector<long>>* inpt);

/**
 * @brief Constructs `neutrino` particle objects from tensor-based input data.
 * @details This function takes several tensors containing particle kinematic information and indices,
 *          along with distance information, and uses them to instantiate `neutrino` objects.
 *          It likely processes the input tensor (`inpt`) row by row or element by element, potentially using
 *          the lepton (`ln`) and b-quark (`bn`) index tensors to associate specific data points with
 *          leptons and b-quarks relevant to the neutrino reconstruction. The distance vector (`dst`)
 *          might provide additional geometric or event-specific information.
 * @param inpt A pointer to a `torch::Tensor` containing the primary particle data (e.g., momenta, energy).
 *             The structure of this tensor (e.g., shape, columns) is crucial for correct interpretation.
 * @param ln A pointer to a `torch::Tensor` containing indices related to leptons. This can be `nullptr` if
 *           lepton index information is not provided or needed.
 * @param bn A pointer to a `torch::Tensor` containing indices related to b-quarks. This can be `nullptr` if
 *           b-quark index information is not provided or needed.
 * @param dst A pointer to a `std::vector<double>` containing distance values or related metrics, potentially
 *            one value per event or particle.
 * @return A `std::vector` containing pointers to the newly constructed `neutrino` objects. The caller
 *         is typically responsible for managing the memory of these dynamically allocated objects.
 *         Returns an empty vector if construction fails or no particles are generated.
 */
std::vector<neutrino*> construct_particle(torch::Tensor* inpt, torch::Tensor* ln, torch::Tensor* bn, std::vector<double>* dst);

/**
 * @brief Computes neutrino four-momenta solutions for top quark pair (ttbar) decay events.
 * @details This function implements the "NuNu" algorithm, a method designed to solve for the four-momenta
 *          of the two neutrinos produced in the dileptonic decay channel of a top quark pair (t -> Wb -> lvb).
 *          It takes as input the measured four-momenta of the two b-quarks and the two leptons, the missing
 *          transverse energy (MET) vector (magnitude and angle), and expected mass constraints for intermediate
 *          (W boson) or final state (top quark) particles. An optimization procedure is employed to find the
 *          neutrino momenta that best satisfy the kinematic constraints of the decay, considering the MET and
 *          mass hypotheses. The computation can be performed on either CPU or GPU (CUDA).
 *
 * @param pmc_b1 Pointer to a vector of vectors, where each inner vector represents the four-momentum
 *               [px, py, pz, E] of the first b-quark for an event.
 * @param pmc_b2 Pointer to a vector of vectors, representing the four-momenta [px, py, pz, E] of the
 *               second b-quark for each event.
 * @param pmc_l1 Pointer to a vector of vectors, representing the four-momenta [px, py, pz, E] of the
 *               first lepton for each event.
 * @param pmc_l2 Pointer to a vector of vectors, representing the four-momenta [px, py, pz, E] of the
 *               second lepton for each event.
 * @param met Pointer to a vector of doubles, containing the magnitude of the missing transverse energy (MET)
 *            for each event.
 * @param phi Pointer to a vector of doubles, containing the azimuthal angle (phi) of the MET vector
 *            for each event.
 * @param mass1 Pointer to a vector of vectors, where each inner vector contains the mass hypotheses
 *              (e.g., [mass_W, mass_top]) associated with the system involving the first lepton (l1) and b-quark (b1).
 * @param mass2 Pointer to a vector of vectors, containing the mass hypotheses (e.g., [mass_W, mass_top])
 *              associated with the system involving the second lepton (l2) and b-quark (b2).
 * @param dev A string specifying the computation device: "cpu" or "cuda".
 * @param null A double value used to represent invalid or missing results within the output neutrino objects.
 * @param step A double representing the step size or learning rate used in the internal optimization algorithm.
 * @param tolerance A double defining the convergence criterion for the optimization algorithm. The optimization
 *                  stops when the change between iterations is below this threshold.
 * @param timeout An unsigned integer specifying the maximum number of iterations allowed for the optimization
 *                algorithm before it terminates (even if convergence is not reached).
 *
 * @return A `std::vector` where each element is a `std::pair` of pointers (`neutrino*`, `neutrino*`).
 *         Each pair represents the computed solution for the first neutrino (associated with l1, b1) and the
 *         second neutrino (associated with l2, b2) for a corresponding input event. The caller is responsible
 *         for managing the memory of the pointed-to `neutrino` objects. Returns an empty vector if the
 *         computation encounters an error or if no solutions are found.
 */
std::vector<std::pair<neutrino*, neutrino*>> pyc::nusol::NuNu(
    std::vector<std::vector<double>>* pmc_b1, std::vector<std::vector<double>>* pmc_b2,
    std::vector<std::vector<double>>* pmc_l1, std::vector<std::vector<double>>* pmc_l2,
    std::vector<double>* met, std::vector<double>* phi,
    std::vector<std::vector<double>>* mass1, std::vector<std::vector<double>>* mass2,
    std::string dev, const double null, const double step, const double tolerance, const unsigned int timeout
);

/**
 * @brief Reconstructs neutrino pairs using a combinatorial approach based on kinematic inputs and topological constraints.
 * @details This function implements a combinatorial solver to determine possible neutrino four-momenta in particle physics events,
 *          likely focusing on scenarios like ttbar dileptonic decays. It differs from the `NuNu` function by potentially exploring
 *          different combinations or assignments of particles if the input structure allows for ambiguity. It utilizes measured
 *          kinematics (MET, visible particle momenta) and topological information (which particles are b-jets, which are leptons,
 *          potential associations) along with mass constraints (top quark mass `mT`, W boson mass `mW`) to constrain the possible
 *          neutrino solutions. The solver iterates through possibilities (`steps`) and may apply perturbations (`perturb`) to explore
 *          the solution space. The computation leverages PyTorch for efficiency and can run on CPU or GPU.
 *
 * @param met_ Pointer to a vector of doubles representing the magnitude of the missing transverse energy (MET) for each event.
 * @param phi_ Pointer to a vector of doubles representing the azimuthal angle (phi) of the MET for each event.
 * @param pmc_ Pointer to a vector of vectors of doubles. This likely represents the four-momenta [px, py, pz, E] of all relevant
 *             reconstructed particles (leptons, jets) in the event. The interpretation depends on the associated index vectors.
 * @param bth_ Pointer to a vector of longs. These indices likely map elements in `pmc_` to identified b-hadrons or b-jets.
 * @param is_b_ Pointer to a vector of longs, potentially indicating for each particle in `pmc_` whether it's associated with a b-hadron
 *              (e.g., a lepton originating from a b-decay).
 * @param is_l_ Pointer to a vector of longs, indicating for each particle in `pmc_` whether it is identified as a lepton.
 * @param dev A string specifying the computation device: "cpu" or "cuda".
 * @param mT The assumed mass of the top quark (double), used as a constraint in the reconstruction.
 * @param mW The assumed mass of the W boson (double), used as a constraint in the reconstruction.
 * @param null A double value used internally or to mark invalid results.
 * @param perturb A double value representing a perturbation factor, possibly used to explore solutions around initial guesses or
 *                to handle numerical instabilities.
 * @param steps A long integer specifying the number of steps, iterations, or combinations to explore in the combinatorial search.
 *              A value of zero might lead to immediate failure or an empty result.
 *
 * @return A `std::vector` of `std::pair<neutrino*, neutrino*>`. Each pair contains pointers to the reconstructed neutrino objects
 *         representing a possible solution pair for an event based on the combinatorial analysis. The caller is responsible for
 *         managing the memory of these `neutrino` objects. Returns an empty vector if the computation fails (e.g., `steps` is zero,
 *         incompatible inputs) or if no valid combinatorial solutions are found.
 */
std::vector<std::pair<neutrino*, neutrino*>> pyc::nusol::combinatorial(
    std::vector<double>* met_, std::vector<double>* phi_, std::vector<std::vector<double>>* pmc_,
    std::vector<long>*   bth_,  std::vector<long>* is_b_, std::vector<long>* is_l_, std::string dev,
    double mT, double mW, double null, double perturb, long steps
);
