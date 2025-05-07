/**
 * @brief Calculates the required number of blocks for a 1D CUDA grid configuration.
 * @details This function determines the number of blocks needed to process a total
 *          number of elements (`lx`) given a specific number of threads per block (`thl`).
 *          It computes `ceil(lx / thl)` using integer arithmetic to ensure enough
 *          blocks are allocated to cover all elements. This is a fundamental calculation
 *          for setting up CUDA kernel launch configurations.
 *
 * @param[in] lx The total number of elements or tasks to be processed in the dimension.
 * @param[in] thl The number of threads configured for each block in that dimension.
 *
 * @return unsigned int The calculated number of blocks required for the grid's dimension.
 *         This value ensures that `num_blocks * thl >= lx`.
 *
 * @note This function assumes a 1D distribution of work. For multi-dimensional grids,
 *       it should be called for each dimension separately.
 * @see blk_()
 */
unsigned int blkn(unsigned int lx, int thl);

/**
 * @brief Creates a 1-dimensional CUDA block configuration (`dim3`).
 * @details This function simplifies the creation of a `dim3` structure representing the
 *          grid dimensions (number of blocks) for launching a 1D CUDA kernel. It uses the
 *          `blkn` function to calculate the required number of blocks in the x-dimension
 *          based on the total size (`dx`) and threads per block (`thrx`). The y and z
 *          dimensions of the grid are set to 1.
 *
 * @param[in] dx The total size (number of elements or tasks) in the x-dimension.
 * @param[in] thrx The number of threads per block configured for the x-dimension.
 *
 * @return const dim3 A `dim3` structure configured with `(blkn(dx, thrx), 1, 1)`,
 *         suitable for the first argument of a CUDA kernel launch `<<<...>>>`.
 *
 * @see blkn()
 */
const dim3 blk_(unsigned int dx, int thrx);

/**
 * @brief Creates a 2-dimensional CUDA block configuration (`dim3`).
 * @details This function facilitates the creation of a `dim3` structure for launching
 *          a 2D CUDA kernel. It calculates the required number of blocks in both the
 *          x and y dimensions using the `blkn` function based on the respective total
 *          sizes (`dx`, `dy`) and threads per block (`thrx`, `thry`). The z dimension
 *          of the grid is set to 1.
 *
 * @param[in] dx The total size (number of elements or tasks) in the x-dimension.
 * @param[in] thrx The number of threads per block configured for the x-dimension.
 * @param[in] dy The total size (number of elements or tasks) in the y-dimension.
 * @param[in] thry The number of threads per block configured for the y-dimension.
 *
 * @return const dim3 A `dim3` structure configured with
 *         `(blkn(dx, thrx), blkn(dy, thry), 1)`, suitable for the first argument
 *         of a CUDA kernel launch `<<<...>>>`.
 *
 * @see blkn()
 */
const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry);

/**
 * @brief Creates a 3-dimensional CUDA block configuration (`dim3`).
 * @details This function assists in creating a `dim3` structure for launching a 3D
 *          CUDA kernel. It computes the required number of blocks for the x, y, and z
 *          dimensions using the `blkn` function, based on the total sizes (`dx`, `dy`, `dz`)
 *          and threads per block (`thrx`, `thry`, `thrz`) specified for each dimension.
 *
 * @param[in] dx The total size (number of elements or tasks) in the x-dimension.
 * @param[in] thrx The number of threads per block configured for the x-dimension.
 * @param[in] dy The total size (number of elements or tasks) in the y-dimension.
 * @param[in] thry The number of threads per block configured for the y-dimension.
 * @param[in] dz The total size (number of elements or tasks) in the z-dimension.
 * @param[in] thrz The number of threads per block configured for the z-dimension.
 *
 * @return const dim3 A `dim3` structure configured with
 *         `(blkn(dx, thrx), blkn(dy, thry), blkn(dz, thrz))`, suitable for the
 *         first argument of a CUDA kernel launch `<<<...>>>`.
 *
 * @see blkn()
 */
const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry, unsigned int dz, int thrz);

/**
 * @brief Splits a given string into a vector of substrings based on a specified delimiter.
 * @details This utility function takes an input string (`inpt`) and divides it into
 *          multiple substrings wherever the `search` delimiter string occurs. The resulting
 *          substrings (excluding the delimiters themselves) are stored and returned in a
 *          `std::vector<std::string>`. If the delimiter is not found, the vector will
 *          contain the original string as its only element. If the delimiter appears
 *          consecutively, empty strings may be included in the result.
 *
 * @param[in] inpt The `std::string` to be split.
 * @param[in] search The `std::string` used as the delimiter for splitting.
 *
 * @return std::vector<std::string> A vector containing the substrings resulting from the split.
 *         The order of substrings corresponds to their appearance in the original string.
 */
std::vector<std::string> split(std::string inpt, std::string search);

/**
 * @brief Moves a PyTorch tensor to a specified target device (CPU or CUDA).
 * @details This function takes a pointer to a PyTorch tensor (`inx`) and a device
 *          identifier string (`dev`) and returns a *new* tensor containing the same data
 *          but residing on the specified device. The original tensor pointed to by `inx`
 *          remains unchanged on its original device. The device string should follow the
 *          PyTorch format, e.g., "cuda:0", "cuda:1", "cpu". This internally calls the
 *          tensor's `.to()` method.
 *
 * @param[in] dev A `std::string` specifying the target device (e.g., "cuda:0", "cpu").
 * @param[in] inx A pointer to the `torch::Tensor` to be moved. The tensor itself is not
 *                modified.
 *
 * @return torch::Tensor A new tensor with the same data as the input tensor, but located
 *         on the device specified by `dev`.
 *
 * @note The function returns a new tensor instance. The caller is responsible for managing
 *       the lifetime of this new tensor. The original tensor is unaffected.
 * @see changedev(torch::Tensor* inpt)
 */
torch::Tensor changedev(std::string dev, torch::Tensor* inx);

/**
 * @brief Sets the active CUDA device context to match the device of a given tensor.
 * @details This function inspects the device where the input tensor (`inpt`) resides
 *          (e.g., CUDA device 0, CUDA device 1) and sets the current CUDA device context
 *          for the calling thread to that specific device using `c10::cuda::set_device`.
 *          This is crucial for ensuring that subsequent CUDA operations (like kernel
 *          launches or CUDA API calls) are executed on the same device as the tensor,
 *          preventing cross-device errors.
 *
 * @param[in] inpt A pointer to the `torch::Tensor` whose device will determine the
 *                 active CUDA context. The tensor itself is not modified.
 *
 * @note This function has no effect if the input tensor is on the CPU. It only changes
 *       the active CUDA device if the tensor resides on a CUDA device.
 * @see changedev(std::string dev, torch::Tensor* inx)
 * @see c10::cuda::set_device()
 */
void changedev(torch::Tensor* inpt);

/**
 * @brief Creates `torch::TensorOptions` configured with the data type and device of a reference tensor.
 * @details This utility function simplifies the creation of new tensors that need to have
 *          the same properties (specifically, data type and device) as an existing tensor.
 *          It takes a pointer to a reference tensor (`v`) and returns a `torch::TensorOptions`
 *          object pre-configured with the `dtype` and `device` of that tensor. This options
 *          object can then be used directly when constructing new tensors (e.g., `torch::zeros`,
 *          `torch::ones`, `torch::empty`).
 *
 * @param[in] v A pointer to the `torch::Tensor` whose data type and device will be used
 *              to configure the options. The tensor itself is not modified.
 *
 * @return torch::TensorOptions A `torch::TensorOptions` object initialized with the
 *         `dtype` and `device` properties copied from the tensor pointed to by `v`.
 *
 * @see torch::Tensor::options()
 */
torch::TensorOptions MakeOp(torch::Tensor* v);

/**
 * @brief Reshapes a tensor to specified dimensions and ensures it is memory-contiguous.
 * @details This function takes a pointer to a tensor (`inpt`) and reshapes it according
 *          to the dimensions provided in the `dim` vector. Crucially, it also ensures
 *          that the resulting tensor is contiguous in memory by calling `.contiguous()`.
 *          Memory contiguity is often a requirement for efficient operations, especially
 *          when passing tensor data to CUDA kernels or external libraries. The function
 *          returns a *new* tensor with the desired shape and guaranteed contiguity.
 *
 * @param[in] inpt A pointer to the `torch::Tensor` to be reshaped and made contiguous.
 *                 The original tensor is not modified.
 * @param[in] dim A `std::vector<signed long>` representing the desired target shape
 *                for the tensor. The total number of elements must remain consistent
 *                with the original tensor.
 *
 * @return torch::Tensor A new tensor that has been reshaped to the dimensions specified
 *         by `dim` and is guaranteed to be contiguous in memory.
 *
 * @note This function returns a new tensor instance. If the input tensor was already
 *       contiguous and had the correct shape, the returned tensor might share the
 *       same underlying data storage; otherwise, a data copy might occur during the
 *       `.contiguous()` call.
 * @see torch::Tensor::reshape(at::IntArrayRef)
 * @see torch::Tensor::contiguous()
 * @see format(std::vector<torch::Tensor> v, std::vector<signed long> dim)
 */
torch::Tensor format(torch::Tensor* inpt, std::vector<signed long> dim);

/**
 * @brief Reshapes multiple tensors to a common shape and concatenates them along the first dimension.
 * @details This function processes a vector of input tensors (`v`). Each tensor in the
 *          vector is first reshaped to the target dimensions specified by `dim` and made
 *          contiguous using the `format(torch::Tensor*, std::vector<signed long>)` helper
 *          function. Subsequently, all the reshaped, contiguous tensors are concatenated
 *          together along dimension 0 using `torch::cat`. This is useful for batching or
 *          combining multiple processed tensors into a single larger tensor.
 *
 * @param[in] v A `std::vector<torch::Tensor>` containing the input tensors to be processed
 *              and concatenated. The original tensors within the vector are not modified.
 * @param[in] dim A `std::vector<signed long>` representing the target shape to which *each*
 *                tensor in the input vector `v` will be reshaped before concatenation.
 *                The total number of elements for each input tensor must be compatible
 *                with this shape.
 *
 * @return torch::Tensor A single tensor resulting from reshaping each input tensor to `dim`,
 *         making them contiguous, and then concatenating them along the first dimension (dim=0).
 *         The resulting tensor will have a shape like `(N * dim[0], dim[1], ..., dim[k])`
 *         where `N` is the number of tensors in the input vector `v`.
 *
 * @note Assumes all tensors in the input vector `v` can be reshaped to the dimensions `dim`.
 *       The concatenation happens along the 0-th dimension.
 * @see format(torch::Tensor* inpt, std::vector<signed long> dim)
 * @see torch::cat(at::TensorList, int64_t)
 */
torch::Tensor format(std::vector<torch::Tensor> v, std::vector<signed long> dim);
