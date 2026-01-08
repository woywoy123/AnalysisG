/**
 * @brief Computes the cofactor of a specific element in a 2x2 matrix represented by a C-style array.
 * @details This function calculates the cofactor for the element at the specified row and column indices within the input matrix `_M`.
 *          The cofactor is the determinant of the submatrix obtained by removing the specified row and column, multiplied by a sign factor (-1)^(row + col).
 *          The `cf` parameter controls whether this sign factor is applied.
 * @tparam scalar_t The data type of the matrix elements (e.g., float, double). This type must support arithmetic operations.
 * @tparam size_x1 The size of the first dimension (rows) of the input matrix `_M`. Expected to be 2 for a 2x2 matrix cofactor calculation.
 * @tparam size_y1 The size of the second dimension (columns) of the input matrix `_M`. Expected to be 2 for a 2x2 matrix cofactor calculation.
 * @param _M A reference to the input 2D C-style array representing the matrix. The function reads elements from this matrix.
 * @param _idy The row index (0-based) of the element for which the cofactor is to be calculated.
 * @param _idz The column index (0-based) of the element for which the cofactor is to be calculated.
 * @param cf A boolean flag. If true, the standard cofactor sign (-1)^(_idy + _idz) is applied. If false, only the determinant of the submatrix is returned.
 * @return The calculated cofactor value as a `scalar_t`.
 */
template <typename scalar_t, size_t size_x1, size_t size_y1>
__device__ scalar_t _cofactor(scalar_t (&_M)[size_x1][size_y1], const unsigned int _idy, unsigned int _idz, bool cf);

/**
 * @brief Calculates the reciprocal (1/p) of the value pointed to by `p`.
 * @details This function computes the multiplicative inverse of the scalar value located at the memory address `p`.
 *          It includes a check to prevent division by zero. If the input value is zero or very close to zero (within floating-point precision limits),
 *          it returns 0.0 to avoid infinity or NaN results.
 * @tparam scalar_t The data type of the scalar value (e.g., float, double). Must support division and comparison.
 * @param p A pointer to the scalar value whose reciprocal is needed. The function dereferences this pointer to get the value.
 * @return The reciprocal of the value pointed to by `p` as a `scalar_t`. Returns 0.0 if the input value is zero.
 */
template <typename scalar_t>
__device__ scalar_t _div(scalar_t* p);

/**
 * @brief Calculates the reciprocal (1/p) of the given scalar value `p`.
 * @details This function computes the multiplicative inverse of the scalar value `p`.
 *          It includes a check to prevent division by zero. If the input value `p` is zero or very close to zero (within floating-point precision limits),
 *          it returns 0.0 to avoid infinity or NaN results.
 * @tparam scalar_t The data type of the scalar value (e.g., float, double). Must support division and comparison.
 * @param p The scalar value whose reciprocal is needed.
 * @return The reciprocal of `p` as a `scalar_t`. Returns 0.0 if `p` is zero.
 */
template <typename scalar_t>
__device__ scalar_t _div(scalar_t p);

/**
 * @brief Calculates the square (p*p) of the value pointed to by `p`.
 * @details This function computes the square of the scalar value located at the memory address `p`.
 *          It dereferences the pointer `p` and multiplies the value by itself.
 * @tparam scalar_t The data type of the scalar value (e.g., float, double). Must support multiplication.
 * @param p A pointer to the scalar value to be squared. The function dereferences this pointer.
 * @return The square of the value pointed to by `p` as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t _p2(scalar_t* p);

/**
 * @brief Clips a scalar value `p` to a small positive value if it is close to zero.
 * @details This function checks if the absolute value of the input scalar `p` is below a small threshold (1e-14).
 *          If it is, the function returns the threshold value (1e-14). Otherwise, it returns the original value `p`.
 *          This is often used to avoid issues with subsequent operations like division or logarithms involving very small numbers.
 * @tparam scalar_t The data type of the scalar value (e.g., float, double). Must support absolute value and comparison.
 * @param p The scalar value to be clipped.
 * @return The original value `p` if its absolute value is greater than or equal to 1e-14, otherwise returns 1e-14. The return type is `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t _clp(scalar_t p);

/**
 * @brief Calculates the square root of the value pointed to by `p`, handling negative inputs.
 * @details This function computes the square root of the scalar value located at the memory address `p`.
 *          If the value is non-negative, it returns the standard positive square root.
 *          If the value is negative, it returns the negative of the square root of the absolute value (-sqrt(|p|)).
 *          This behavior might be specific to a particular physical context or numerical algorithm.
 * @tparam scalar_t The data type of the scalar value (e.g., float, double). Must support square root, comparison, and negation.
 * @param p A pointer to the scalar value whose square root is needed. The function dereferences this pointer.
 * @return The square root of the value pointed to by `p` as a `scalar_t`. Returns `sqrt(p)` if `p >= 0`, and `-sqrt(-p)` if `p < 0`.
 */
template <typename scalar_t>
__device__ scalar_t _sqrt(scalar_t* p);

/**
 * @brief Calculates the square root of the scalar value `p`, handling negative inputs.
 * @details This function computes the square root of the scalar value `p`.
 *          If `p` is non-negative, it returns the standard positive square root (`sqrt(p)`).
 *          If `p` is negative, it returns the negative of the square root of its absolute value (`-sqrt(-p)`).
 *          This behavior might be specific to a particular physical context or numerical algorithm where the sign encodes information.
 * @tparam scalar_t The data type of the scalar value (e.g., float, double). Must support square root, comparison, and negation.
 * @param p The scalar value whose square root is needed.
 * @return The square root of `p` as a `scalar_t`. Returns `sqrt(p)` if `p >= 0`, and `-sqrt(-p)` if `p < 0`.
 */
template <typename scalar_t>
__device__ scalar_t _sqrt(scalar_t p);

/**
 * @brief Performs a comparison based on three scalar values, likely related to matrix elements or vector components.
 * @details This function calculates `sqrt(xx*yy - xy*xy)`. It appears to compute a quantity related to the determinant or a similar metric involving squares and cross-terms.
 *          The exact interpretation depends on the context where `xx`, `yy`, and `xy` originate (e.g., components of a covariance matrix).
 *          It uses the custom `_sqrt` function, which handles potentially negative arguments inside the square root based on its specific definition.
 * @tparam scalar_t The data type of the scalar values (e.g., float, double). Must support multiplication, subtraction, and the custom `_sqrt` operation.
 * @param xx The first scalar value, likely representing a squared term or diagonal element.
 * @param yy The second scalar value, likely representing another squared term or diagonal element.
 * @param xy The third scalar value, likely representing a cross-term or off-diagonal element.
 * @return The result of `_sqrt(xx * yy - xy * xy)` as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t _cmp(scalar_t xx, scalar_t yy, scalar_t xy);

/**
 * @brief Calculates the arccosine of a ratio derived from two scalar values pointed to by `sm` and `_pz`.
 * @details This function computes `acos(*_pz / *sm)`. It first dereferences `_pz` and `sm` to get the scalar values.
 *          It then calculates their ratio and computes the arccosine of this ratio.
 *          The function includes clipping of the ratio to the valid domain of `acos` ([-1, 1]) to prevent domain errors.
 *          It also uses the custom `_div` function for the division, which handles potential division by zero in `*sm`.
 * @tparam scalar_t The data type of the scalar values (e.g., float, double). Must support division (`_div`), comparison, and `acos`.
 * @param sm A pointer to the first scalar value (likely the denominator).
 * @param _pz A pointer to the second scalar value (likely the numerator).
 * @return The arccosine of the ratio `*_pz / *sm`, clipped to the range [0, PI], as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t _arccos(scalar_t* sm, scalar_t* _pz);

/**
 * @brief Calculates a custom modulo operation for the value pointed to by `diff`.
 * @details This function adjusts the value pointed to by `diff` to lie within the range (-PI, PI].
 *          It repeatedly adds or subtracts 2*PI until the value falls within the desired interval.
 *          This is commonly used for normalizing angles, particularly in physics or robotics.
 * @tparam scalar_t The data type of the scalar value (e.g., float, double). Must support comparison, addition, and subtraction. Constants like PI must be defined for this type.
 * @param diff A pointer to the scalar value (e.g., an angle difference) to be normalized. The function modifies the value in place indirectly via the pointer and also returns the normalized value.
 * @return The normalized value (within (-PI, PI]) as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t minus_mod(scalar_t* diff);

/**
 * @brief Calculates the x-component of momentum (px) given transverse momentum (pt) and azimuthal angle (phi).
 * @details This function computes `px = pt * cos(phi)`. It dereferences the pointers `_pt` and `_phi` to get the values of transverse momentum and azimuthal angle, respectively.
 *          It then applies the standard formula to find the Cartesian x-component from polar coordinates in the transverse plane.
 * @tparam scalar_t The data type for momentum and angle (e.g., float, double). Must support multiplication and `cos`.
 * @param _pt A pointer to the transverse momentum value.
 * @param _phi A pointer to the azimuthal angle value (in radians).
 * @return The x-component of momentum (`px`) as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t px_(scalar_t* _pt, scalar_t* _phi);

/**
 * @brief Calculates the y-component of momentum (py) given transverse momentum (pt) and azimuthal angle (phi).
 * @details This function computes `py = pt * sin(phi)`. It dereferences the pointers `_pt` and `_phi` to get the values of transverse momentum and azimuthal angle, respectively.
 *          It then applies the standard formula to find the Cartesian y-component from polar coordinates in the transverse plane.
 * @tparam scalar_t The data type for momentum and angle (e.g., float, double). Must support multiplication and `sin`.
 * @param _pt A pointer to the transverse momentum value.
 * @param _phi A pointer to the azimuthal angle value (in radians).
 * @return The y-component of momentum (`py`) as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t py_(scalar_t* _pt, scalar_t* _phi);

/**
 * @brief Calculates the z-component of momentum (pz) given transverse momentum (pt) and pseudorapidity (eta).
 * @details This function computes `pz = pt * sinh(eta)`. It dereferences the pointers `_pt` and `_eta` to get the values of transverse momentum and pseudorapidity, respectively.
 *          It then applies the standard formula relating pz, pt, and eta in high-energy physics.
 * @tparam scalar_t The data type for momentum and pseudorapidity (e.g., float, double). Must support multiplication and `sinh`.
 * @param _pt A pointer to the transverse momentum value.
 * @param _eta A pointer to the pseudorapidity value.
 * @return The z-component of momentum (`pz`) as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t pz_(scalar_t* _pt, scalar_t* _eta);

/**
 * @brief Calculates the transverse momentum (pt) given the x and y components of momentum (px, py).
 * @details This function computes `pt = sqrt(px*px + py*py)`. It dereferences the pointers `_px` and `_py` to get the momentum components.
 *          It calculates the sum of their squares and then takes the square root to find the magnitude of the momentum vector in the transverse (x-y) plane.
 * @tparam scalar_t The data type for momentum components (e.g., float, double). Must support multiplication, addition, and `sqrt`.
 * @param _px A pointer to the x-component of momentum.
 * @param _py A pointer to the y-component of momentum.
 * @return The transverse momentum (`pt`) as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t pt_(scalar_t* _px, scalar_t* _py);

/**
 * @brief Calculates the pseudorapidity (eta) given the Cartesian components of momentum (px, py, pz).
 * @details This function computes `eta = asinh(pz / pt)`, where `pt = sqrt(px*px + py*py)`.
 *          It first calculates the transverse momentum `pt` using the `pt_` helper function with the values pointed to by `_px` and `_py`.
 *          Then, it calculates the ratio of the z-component of momentum (from `*_pz`) to the transverse momentum `pt`.
 *          Finally, it computes the inverse hyperbolic sine (`asinh`) of this ratio. Includes checks for `pt` being close to zero.
 * @tparam scalar_t The data type for momentum components (e.g., float, double). Must support operations used in `pt_`, division, and `asinh`.
 * @param _px A pointer to the x-component of momentum.
 * @param _py A pointer to the y-component of momentum.
 * @param _pz A pointer to the z-component of momentum.
 * @return The pseudorapidity (`eta`) as a `scalar_t`. Returns 0 if `pt` is zero.
 */
template <typename scalar_t>
__device__ scalar_t eta_(scalar_t* _px, scalar_t* _py, scalar_t* _pz);

/**
 * @brief Calculates the pseudorapidity (eta) given transverse momentum (pt) and the z-component of momentum (pz).
 * @details This function computes `eta = asinh(pz / pt)`. It dereferences the pointers `_pt` and `_pz` to get the values.
 *          It calculates the ratio `pz / pt` using the custom `_div` function (which handles potential division by zero if `pt` is zero).
 *          Then, it computes the inverse hyperbolic sine (`asinh`) of this ratio.
 * @tparam scalar_t The data type for momentum components (e.g., float, double). Must support division (`_div`) and `asinh`.
 * @param _pt A pointer to the transverse momentum value.
 * @param _pz A pointer to the z-component of momentum value.
 * @return The pseudorapidity (`eta`) as a `scalar_t`. Returns 0 if `pt` is zero (handled by `_div`).
 */
template <typename scalar_t>
__device__ scalar_t eta_(scalar_t* _pt, scalar_t* _pz);

/**
 * @brief Calculates the azimuthal angle (phi) given the x and y components of momentum (px, py).
 * @details This function computes `phi = atan2(py, px)`. It dereferences the pointers `_px` and `_py` to get the momentum components.
 *          It then uses the `atan2` function, which correctly handles the signs of `px` and `py` to return an angle in the range (-PI, PI], representing the angle in the transverse (x-y) plane.
 * @tparam scalar_t The data type for momentum components (e.g., float, double). Must support `atan2`.
 * @param _px A pointer to the x-component of momentum.
 * @param _py A pointer to the y-component of momentum.
 * @return The azimuthal angle (`phi`) in radians, in the range (-PI, PI], as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t phi_(scalar_t* _px, scalar_t* _py);

/**
 * @brief Calculates an element of a 3x3 rotation matrix for rotation around the x-axis.
 * @details This function returns the element at row `_idy` and column `_idz` of a standard 3D rotation matrix corresponding to a rotation by angle `*_a` around the x-axis.
 *          The matrix is:
 *          [[ 1,  0,       0      ],
 *           [ 0,  cos(a), -sin(a) ],
 *           [ 0,  sin(a),  cos(a) ]]
 *          It uses helper functions `foptim` and `trigger` to efficiently select the correct value (1, 0, +/-sin(a), +/-cos(a)) based on the indices.
 * @tparam scalar_t The data type for the angle and matrix elements (e.g., float, double). Must support `cos`, `sin`, negation.
 * @param _a A pointer to the rotation angle (in radians).
 * @param _idy The row index (0, 1, or 2) of the desired matrix element.
 * @param _idz The column index (0, 1, or 2) of the desired matrix element.
 * @return The value of the rotation matrix element at (`_idy`, `_idz`) as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t _rx(scalar_t* _a, const unsigned int _idy, const unsigned int _idz);

/**
 * @brief Calculates an element of a 3x3 rotation matrix for rotation around the y-axis.
 * @details This function returns the element at row `_idy` and column `_idz` of a standard 3D rotation matrix corresponding to a rotation by angle `*_a` around the y-axis.
 *          The matrix is:
 *          [[ cos(a),  0,  sin(a) ],
 *           [ 0,       1,  0      ],
 *           [-sin(a),  0,  cos(a) ]]
 *          It uses helper functions `foptim` and `trigger` to efficiently select the correct value (1, 0, +/-sin(a), +/-cos(a)) based on the indices.
 * @tparam scalar_t The data type for the angle and matrix elements (e.g., float, double). Must support `cos`, `sin`, negation.
 * @param _a A pointer to the rotation angle (in radians).
 * @param _idy The row index (0, 1, or 2) of the desired matrix element.
 * @param _idz The column index (0, 1, or 2) of the desired matrix element.
 * @return The value of the rotation matrix element at (`_idy`, `_idz`) as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t _ry(scalar_t* _a, const unsigned int _idy, const unsigned int _idz);

/**
 * @brief Calculates an element of a 3x3 rotation matrix for rotation around the z-axis.
 * @details This function returns the element at row `_idy` and column `_idz` of a standard 3D rotation matrix corresponding to a rotation by angle `*_a` around the z-axis.
 *          The matrix is:
 *          [[ cos(a), -sin(a), 0 ],
 *           [ sin(a),  cos(a), 0 ],
 *           [ 0,       0,      1 ]]
 *          It uses helper functions `foptim` and `trigger` to efficiently select the correct value (1, 0, +/-sin(a), +/-cos(a)) based on the indices.
 * @tparam scalar_t The data type for the angle and matrix elements (e.g., float, double). Must support `cos`, `sin`, negation.
 * @param _a A pointer to the rotation angle (in radians).
 * @param _idy The row index (0, 1, or 2) of the desired matrix element.
 * @param _idz The column index (0, 1, or 2) of the desired matrix element.
 * @return The value of the rotation matrix element at (`_idy`, `_idz`) as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t _rz(scalar_t* _a, const unsigned int _idy, const unsigned int _idz);

/**
 * @brief Calculates the dot product of a specific row of matrix `v1` and a specific column of matrix `v2`.
 * @details This function computes the sum of the element-wise products between the elements of row `row` in matrix `v1` and the elements of column `col` in matrix `v2`.
 *          The summation proceeds for `dx` elements. It assumes that the inner dimensions match for this operation to be mathematically valid (i.e., `size_y1 >= dx` and `size_x2 >= dx`).
 *          Specifically, it calculates: `sum(v1[row][k] * v2[k][col] for k from 0 to dx-1)`.
 * @tparam scalar_t The data type of the matrix elements (e.g., float, double). Must support multiplication and addition.
 * @tparam size_x1 The size of the first dimension (rows) of matrix `v1`.
 * @tparam size_y1 The size of the second dimension (columns) of matrix `v1`.
 * @tparam size_x2 The size of the first dimension (rows) of matrix `v2`.
 * @tparam size_y2 The size of the second dimension (columns) of matrix `v2`.
 * @param v1 A reference to the first input 2D C-style array (matrix).
 * @param v2 A reference to the second input 2D C-style array (matrix).
 * @param row The 0-based index of the row to use from matrix `v1`.
 * @param col The 0-based index of the column to use from matrix `v2`.
 * @param dx The number of elements to include in the dot product calculation (typically the inner dimension of the matrix multiplication).
 * @return The dot product result as a `scalar_t`.
 */
template <typename scalar_t, size_t size_x1, size_t size_y1, size_t size_x2, size_t size_y2>
__device__ scalar_t _dot(
    scalar_t (&v1)[size_x1][size_y1], scalar_t (&v2)[size_x2][size_y2],
    const unsigned int row, unsigned int col, unsigned int dx
);

/**
 * @brief Calculates the dot product of the first `dx` elements of two vectors `v1` and `v2`.
 * @details This function computes the sum of the element-wise products of the first `dx` elements of the two input vectors (C-style arrays) `v1` and `v2`.
 *          It calculates: `sum(v1[k] * v2[k] for k from 0 to dx-1)`.
 *          It assumes that both `v1` and `v2` have at least `dx` elements (i.e., `size_x1 >= dx` and `size_x2 >= dx`).
 * @tparam scalar_t The data type of the vector elements (e.g., float, double). Must support multiplication and addition.
 * @tparam size_x1 The declared size of the first input vector `v1`.
 * @tparam size_x2 The declared size of the second input vector `v2`.
 * @param v1 A reference to the first input 1D C-style array (vector).
 * @param v2 A reference to the second input 1D C-style array (vector).
 * @param dx The number of elements from the start of each vector to include in the dot product.
 * @return The dot product of the first `dx` elements as a `scalar_t`.
 */
template <typename scalar_t, size_t size_x1, size_t size_x2>
__device__ scalar_t _dot(scalar_t (&v1)[size_x1], scalar_t (&v2)[size_x2], unsigned int dx);

/**
 * @brief Calculates the dot product of elements within a specified range [`ds`, `de`) for two vectors `v1` and `v2`.
 * @details This function computes the sum of the element-wise products of the elements of the two input vectors (C-style arrays) `v1` and `v2`, starting from index `ds` up to (but not including) index `de`.
 *          It calculates: `sum(v1[k] * v2[k] for k from ds to de-1)`.
 *          It assumes that both `v1` and `v2` have at least `de` elements (i.e., `size_x1 >= de` and `size_x2 >= de`) and that `ds <= de`.
 * @tparam scalar_t The data type of the vector elements (e.g., float, double). Must support multiplication and addition.
 * @tparam size_x1 The declared size of the first input vector `v1`.
 * @tparam size_x2 The declared size of the second input vector `v2`.
 * @param v1 A reference to the first input 1D C-style array (vector).
 * @param v2 A reference to the second input 1D C-style array (vector).
 * @param ds The starting index (inclusive) for the dot product calculation.
 * @param de The ending index (exclusive) for the dot product calculation.
 * @return The dot product over the specified range [`ds`, `de`) as a `scalar_t`.
 */
template <typename scalar_t, size_t size_x1, size_t size_x2>
__device__ scalar_t _dot(scalar_t (&v1)[size_x1], scalar_t (&v2)[size_x2], unsigned int ds, unsigned int de);

/**
 * @brief Calculates the sum of the first `dx` elements of a vector `v1`.
 * @details This function computes the sum of the first `dx` elements of the input vector (C-style array) `v1`.
 *          It calculates: `sum(v1[k] for k from 0 to dx-1)`.
 *          It assumes that `v1` has at least `dx` elements (i.e., `size_x1 >= dx`).
 * @tparam scalar_t The data type of the vector elements (e.g., float, double). Must support addition.
 * @tparam size_x1 The declared size of the input vector `v1`.
 * @param v1 A reference to the input 1D C-style array (vector).
 * @param dx The number of elements from the start of the vector to include in the sum.
 * @return The sum of the first `dx` elements as a `scalar_t`.
 */
template <typename scalar_t, size_t size_x1>
__device__ scalar_t _sum(scalar_t (&v1)[size_x1], const unsigned int dx);

/**
 * @brief Calculates the sum of the first `dx` elements of a vector `v1` (passed as a C-style array without compile-time size).
 * @details This function computes the sum of the first `dx` elements of the input vector (C-style array) `v1`.
 *          It calculates: `sum(v1[k] for k from 0 to dx-1)`.
 *          This version is used when the size of the array is not known at compile time or when passing dynamically sized arrays.
 *          The caller must ensure that `v1` points to a valid memory region containing at least `dx` elements.
 * @tparam scalar_t The data type of the vector elements (e.g., float, double). Must support addition.
 * @param v1 A reference to the input 1D C-style array (vector). The size is not enforced by the template.
 * @param dx The number of elements from the start of the vector to include in the sum.
 * @return The sum of the first `dx` elements as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t _sum(scalar_t (&v1)[], const unsigned int dx);

/**
 * @brief Selects and computes a trigonometric function or a constant based on the integer selector `l`.
 * @details This function acts as a selector for optimized calculations, potentially within rotation matrix computations.
 *          - If `l` is 0, it returns `cos(t)`.
 *          - If `l` is 1, it returns `sin(t)`.
 *          - If `l` is 2, it returns 1.0.
 *          Any other value of `l` results in undefined behavior (though the implementation might default to one case).
 * @tparam scalar_t The data type for the input `t` and the return value (e.g., float, double). Must support `cos`, `sin`, and represent 1.0.
 * @param t The input value (often an angle in radians) for the trigonometric functions.
 * @param l An unsigned integer selector: 0 for cosine, 1 for sine, 2 for constant 1.
 * @return The result of the selected operation (`cos(t)`, `sin(t)`, or 1.0) as a `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t foptim(scalar_t t, const unsigned int l);

/**
 * @brief Selects one of two scalar values based on a boolean condition.
 * @details This function implements a conditional selection, similar to a ternary operator (`con ? v1 : v2`).
 *          If the boolean condition `con` is true, it returns the value `v1`.
 *          If `con` is false, it returns the value `v2`.
 * @tparam scalar_t The data type of the values `v1` and `v2` and the return value (e.g., float, double, int).
 * @param con The boolean condition used for selection.
 * @param v1 The value to return if `con` is true.
 * @param v2 The value to return if `con` is false.
 * @return Returns `v1` if `con` is true, otherwise returns `v2`. The return type is `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t trigger(bool con, scalar_t v1, scalar_t v2);

/**
 * @brief Selects one of three scalar values based on an integer index `dx`.
 * @details This function implements a selection based on the value of the index `dx`.
 *          - If `dx` is 0, it returns `v1`.
 *          - If `dx` is 1, it returns `v2`.
 *          - If `dx` is 2, it returns `v3`.
 *          Any other value of `dx` results in undefined behavior (though the implementation might default to one case, likely `v3`).
 *          This is often used for selecting components or values based on an index.
 * @tparam scalar_t The data type of the values `v1`, `v2`, `v3` and the return value (e.g., float, double, int).
 * @param dx The unsigned integer index used for selection (expected to be 0, 1, or 2).
 * @param v1 The value to return if `dx` is 0.
 * @param v2 The value to return if `dx` is 1.
 * @param v3 The value to return if `dx` is 2.
 * @return Returns `v1`, `v2`, or `v3` based on the value of `dx`. The return type is `scalar_t`.
 */
template <typename scalar_t>
__device__ scalar_t trigger(const unsigned int dx, scalar_t v1, scalar_t v2, scalar_t v3);
