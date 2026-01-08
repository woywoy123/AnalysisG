/**
 * @brief Implementation of Px calculation using a CUDA kernel.
 *
 * Reshapes input tensors `pt` and `phi` to (N, 1) if they are not already.
 * Allocates an output tensor `px_` of shape (N, 1).
 * Launches the `PxK` CUDA kernel to perform the element-wise calculation
 * Px = pt * cos(phi). Block and thread dimensions are determined based on
 * the input size N. Uses `AT_DISPATCH_FLOATING_TYPES` for type handling.
 *
 * @param pt Pointer to a tensor containing transverse momentum values. Expected shape: (N) or (N, 1).
 * @param phi Pointer to a tensor containing azimuthal angle values (in radians). Expected shape: (N) or (N, 1).
 * @return A tensor containing the calculated Px values. Shape: (N, 1).
 * @see PxK
 */
torch::Tensor transform_::Px(torch::Tensor* pt, torch::Tensor* phi);

/**
 * @brief Implementation of Py calculation using a CUDA kernel.
 *
 * Reshapes input tensors `pt` and `phi` to (N, 1) if they are not already.
 * Allocates an output tensor `py_` of shape (N, 1).
 * Launches the `PyK` CUDA kernel to perform the element-wise calculation
 * Py = pt * sin(phi). Block and thread dimensions are determined based on
 * the input size N. Uses `AT_DISPATCH_FLOATING_TYPES` for type handling.
 *
 * @param pt Pointer to a tensor containing transverse momentum values. Expected shape: (N) or (N, 1).
 * @param phi Pointer to a tensor containing azimuthal angle values (in radians). Expected shape: (N) or (N, 1).
 * @return A tensor containing the calculated Py values. Shape: (N, 1).
 * @see PyK
 */
torch::Tensor transform_::Py(torch::Tensor* pt, torch::Tensor* phi);

/**
 * @brief Implementation of Pz calculation using a CUDA kernel.
 *
 * Reshapes input tensors `pt` and `eta` to (N, 1) if they are not already.
 * Allocates an output tensor `pz_` of shape (N, 1).
 * Launches the `PzK` CUDA kernel to perform the element-wise calculation
 * Pz = pt * sinh(eta). Block and thread dimensions are determined based on
 * the input size N. Uses `AT_DISPATCH_FLOATING_TYPES` for type handling.
 *
 * @param pt Pointer to a tensor containing transverse momentum values. Expected shape: (N) or (N, 1).
 * @param eta Pointer to a tensor containing pseudorapidity values. Expected shape: (N) or (N, 1).
 * @return A tensor containing the calculated Pz values. Shape: (N, 1).
 * @see PzK
 */
torch::Tensor transform_::Pz(torch::Tensor* pt, torch::Tensor* eta);

/**
 * @brief Implementation of (Pt, Eta, Phi) -> (Px, Py, Pz) transformation using a CUDA kernel.
 *
 * Expects input `pmu` tensor of shape (N, 3).
 * Allocates an output tensor `pmc` of shape (N, 3).
 * Launches the `PxPyPzK` CUDA kernel, potentially utilizing shared memory (`s`),
 * to perform the transformation for each row. Block and thread dimensions are
 * configured for processing rows and columns. Uses `AT_DISPATCH_FLOATING_TYPES`
 * for type handling.
 *
 * @param pmu Pointer to a 2D tensor with shape (N, 3) representing (Pt, Eta, Phi).
 * @return A 2D tensor containing the transformed (Px, Py, Pz) values. Shape: (N, 3).
 * @see PxPyPzK
 */
torch::Tensor transform_::PxPyPz(torch::Tensor* pmu);

/**
 * @brief Implementation of (Pt, Eta, Phi, [E]) -> (Px, Py, Pz, E) transformation using CUDA kernels.
 *
 * Handles two cases based on the number of columns (`dy`) in the input `pmu` tensor:
 * 1. If `dy >= 4`: Delegates the transformation of the first 3 columns (Pt, Eta, Phi)
 *    to `transform_::PxPyPz(pmu)`, returning a tensor of shape (N, 3) containing (Px, Py, Pz).
 *    Note: This behavior differs from the header documentation; the energy column is *not* copied.
 * 2. If `dy == 3`: Assumes input is (Pt, Eta, Phi). Allocates an output tensor `pmc`
 *    of shape (N, 4). Launches the `PxPyPzEK` CUDA kernel to calculate (Px, Py, Pz, E),
 *    where E is likely calculated assuming massless particles (E = pt * cosh(eta)).
 *    Returns the (N, 4) tensor.
 *
 * Uses `AT_DISPATCH_FLOATING_TYPES` for type handling in the `dy == 3` case.
 *
 * @param pmu Pointer to a 2D tensor with shape (N, >=3) starting with (Pt, Eta, Phi, [E], ...).
 * @return A 2D tensor containing transformed coordinates. Shape is (N, 3) if input columns >= 4,
 *         and (N, 4) if input columns == 3. Columns are (Px, Py, Pz) or (Px, Py, Pz, E).
 * @see PxPyPz(torch::Tensor*)
 * @see PxPyPzEK
 */
torch::Tensor transform_::PxPyPzE(torch::Tensor* pmu);

/**
 * @brief Implementation of Pt calculation using a CUDA kernel.
 *
 * Reshapes input tensors `px` and `py` to (N, 1) if they are not already.
 * Allocates an output tensor `pt_` of shape (N, 1).
 * Launches the `PtK` CUDA kernel to perform the element-wise calculation
 * Pt = sqrt(Px^2 + Py^2). Block and thread dimensions are determined based on
 * the input size N. Uses `AT_DISPATCH_FLOATING_TYPES` for type handling.
 *
 * @param px Pointer to a tensor containing Px values. Expected shape: (N) or (N, 1).
 * @param py Pointer to a tensor containing Py values. Expected shape: (N) or (N, 1).
 * @return A tensor containing the calculated Pt values. Shape: (N, 1).
 * @see PtK
 */
torch::Tensor transform_::Pt(torch::Tensor* px, torch::Tensor* py);

/**
 * @brief Implementation of Phi calculation using a CUDA kernel.
 *
 * Reshapes input tensors `px` and `py` to (N, 1) if they are not already.
 * Allocates an output tensor `phi_` of shape (N, 1).
 * Launches the `PhiK` CUDA kernel to perform the element-wise calculation
 * Phi = atan2(Py, Px). Block and thread dimensions are determined based on
 * the input size N. Uses `AT_DISPATCH_FLOATING_TYPES` for type handling.
 *
 * @param px Pointer to a tensor containing Px values. Expected shape: (N) or (N, 1).
 * @param py Pointer to a tensor containing Py values. Expected shape: (N) or (N, 1).
 * @return A tensor containing the calculated Phi values (in radians). Shape: (N, 1).
 * @see PhiK
 */
torch::Tensor transform_::Phi(torch::Tensor* px, torch::Tensor* py);


/**
 * @brief Implementation of Eta calculation from Cartesian coordinates using a CUDA kernel.
 *
 * Expects input `pmc` tensor of shape (N, >=3) with columns (Px, Py, Pz, ...).
 * Allocates an output tensor `eta` of shape (N, 1).
 * Launches the `EtaK` CUDA kernel to perform the calculation
 * Eta = asinh(Pz / Pt), where Pt = sqrt(Px^2 + Py^2). The kernel handles the Pt=0 case.
 * Block and thread dimensions are determined based on the input size N.
 * Uses `AT_DISPATCH_FLOATING_TYPES` for type handling.
 *
 * @param pmc Pointer to a 2D tensor with shape (N, >=3) representing (Px, Py, Pz, ...).
 * @return A tensor containing the calculated Eta values. Shape: (N, 1).
 * @see EtaK
 */
torch::Tensor transform_::Eta(torch::Tensor* pmc);

/**
 * @brief Implementation of (Px, Py, Pz) -> (Pt, Eta, Phi) transformation using a CUDA kernel.
 *
 * Expects input `pmc` tensor of shape (N, 3).
 * Allocates an output tensor `pmu` of shape (N, 3).
 * Launches the `PtEtaPhiK` CUDA kernel, potentially utilizing shared memory (`s`),
 * to perform the transformation for each row. Block and thread dimensions are
 * configured for processing rows and columns. Uses `AT_DISPATCH_FLOATING_TYPES`
 * for type handling.
 *
 * @param pmc Pointer to a 2D tensor with shape (N, 3) representing (Px, Py, Pz).
 * @return A 2D tensor containing the transformed (Pt, Eta, Phi) values. Shape: (N, 3).
 * @see PtEtaPhiK
 */
torch::Tensor transform_::PtEtaPhi(torch::Tensor* pmc);

/**
 * @brief Implementation of (Px, Py, Pz, [E]) -> (Pt, Eta, Phi, E) transformation using CUDA kernels.
 *
 * Handles two cases based on the number of columns (`dy`) in the input `pmc` tensor:
 * 1. If `dy >= 4`: Delegates the transformation of the first 3 columns (Px, Py, Pz)
 *    to `transform_::PtEtaPhi(pmc)`, returning a tensor of shape (N, 3) containing (Pt, Eta, Phi).
 *    Note: This behavior differs from the header documentation; the energy column is *not* copied.
 * 2. If `dy == 3`: Assumes input is (Px, Py, Pz). Allocates an output tensor `pmu`
 *    of shape (N, 4). Launches the `PtEtaPhiEK` CUDA kernel to calculate (Pt, Eta, Phi, E),
 *    where E is likely calculated assuming massless particles (E = sqrt(Px^2 + Py^2 + Pz^2)).
 *    Returns the (N, 4) tensor.
 *
 * Uses `AT_DISPATCH_FLOATING_TYPES` for type handling in the `dy == 3` case.
 *
 * @param pmc Pointer to a 2D tensor with shape (N, >=3) starting with (Px, Py, Pz, [E], ...).
 * @return A 2D tensor containing transformed coordinates. Shape is (N, 3) if input columns >= 4,
 *         and (N, 4) if input columns == 3. Columns are (Pt, Eta, Phi) or (Pt, Eta, Phi, E).
 * @see PtEtaPhi(torch::Tensor*)
 * @see PtEtaPhiEK
 */
torch::Tensor transform_::PtEtaPhiE(torch::Tensor* pmc);

/**
 * @brief Overload implementation for calculating Eta from individual Px, Py, Pz tensors.
 *
 * Concatenates the input tensors `px`, `py`, `pz` into a single (N, 3) tensor `pmc`
 * using the `format` utility function.
 * Delegates the calculation to the primary `Eta(torch::Tensor* pmc)` function.
 *
 * @param px Pointer to a 1D tensor containing Px values.
 * @param py Pointer to a 1D tensor containing Py values.
 * @param pz Pointer to a 1D tensor containing Pz values.
 * @return A tensor containing the calculated Eta values. Shape: (N, 1).
 * @see Eta(torch::Tensor* pmc)
 * @see format
 */
torch::Tensor transform_::Eta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Overload implementation for transforming individual Px, Py, Pz tensors to (Pt, Eta, Phi).
 *
 * Concatenates the input tensors `px`, `py`, `pz` into a single (N, 3) tensor `pmc`
 * using the `format` utility function.
 * Delegates the transformation to the primary `PtEtaPhi(torch::Tensor* pmc)` function.
 *
 * @param px Pointer to a 1D tensor containing Px values.
 * @param py Pointer to a 1D tensor containing Py values.
 * @param pz Pointer to a 1D tensor containing Pz values.
 * @return A 2D tensor with shape (N, 3) representing momentum in (Pt, Eta, Phi).
 * @see PtEtaPhi(torch::Tensor* pmc)
 * @see format
 */
torch::Tensor transform_::PtEtaPhi(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Overload implementation for transforming individual Px, Py, Pz, E tensors to cylindrical coordinates.
 *
 * Concatenates the input tensors `px`, `py`, `pz`, `e` into a single (N, 4) tensor `pmc`
 * using the `format` utility function.
 * Delegates the transformation to `transform_::PtEtaPhi(torch::Tensor* pmc)`.
 * **Important Note:** This overload calls `PtEtaPhi`, which only processes the first three
 * columns (Px, Py, Pz) and returns an (N, 3) tensor containing (Pt, Eta, Phi). The input
 * energy `e` is effectively ignored in the current implementation.
 *
 * @param px Pointer to a 1D tensor containing Px values.
 * @param py Pointer to a 1D tensor containing Py values.
 * @param pz Pointer to a 1D tensor containing Pz values.
 * @param e Pointer to a 1D tensor containing Energy values.
 * @return A 2D tensor with shape (N, 3) representing momentum in (Pt, Eta, Phi).
 * @see PtEtaPhi(torch::Tensor* pmc)
 * @see format
 */
torch::Tensor transform_::PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Overload implementation for transforming individual Pt, Eta, Phi tensors to (Px, Py, Pz).
 *
 * Concatenates the input tensors `pt`, `eta`, `phi` into a single (N, 3) tensor `pmu`
 * using the `format` utility function.
 * Delegates the transformation to the primary `PxPyPz(torch::Tensor* pmu)` function.
 *
 * @param pt Pointer to a 1D tensor containing Pt values.
 * @param eta Pointer to a 1D tensor containing Eta values.
 * @param phi Pointer to a 1D tensor containing Phi values (in radians).
 * @return A 2D tensor with shape (N, 3) representing momentum in (Px, Py, Pz).
 * @see PxPyPz(torch::Tensor* pmu)
 * @see format
 */
torch::Tensor transform_::PxPyPz(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi);

/**
 * @brief Overload implementation for transforming individual Pt, Eta, Phi, E tensors to Cartesian coordinates.
 *
 * Concatenates the input tensors `pt`, `eta`, `phi`, `energy` into a single (N, 4) tensor `pmu`
 * using the `format` utility function.
 * Delegates the transformation to `transform_::PxPyPz(torch::Tensor* pmu)`.
 * **Important Note:** This overload calls `PxPyPz`, which only processes the first three
 * columns (Pt, Eta, Phi) and returns an (N, 3) tensor containing (Px, Py, Pz). The input
 * energy `energy` is effectively ignored in the current implementation.
 *
 * @param pt Pointer to a 1D tensor containing Pt values.
 * @param eta Pointer to a 1D tensor containing Eta values.
 * @param phi Pointer to a 1D tensor containing Phi values (in radians).
 * @param energy Pointer to a 1D tensor containing Energy values.
 * @return A 2D tensor with shape (N, 3) representing momentum in (Px, Py, Pz).
 * @see PxPyPz(torch::Tensor* pmu)
 * @see format
 */
torch::Tensor transform_::PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* energy);
torch::Tensor transform_::PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* energy){
    torch::Tensor pmu = format({*pt, *eta, *phi, *energy}); 
/**
 * @brief transform_::PxPyPz Funktion
 * 
 * Detaillierte Beschreibung der transform_::PxPyPz Funktion
 */
    return transform_::PxPyPz(&pmu); 
}

