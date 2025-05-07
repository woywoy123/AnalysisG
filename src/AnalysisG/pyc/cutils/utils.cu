/**
 * @file utils.cu
 * @brief Implements utility functions for CUDA operations.
 */

#include <utils/utils.cuh> ///< Includes the header file "utils/utils.cuh", which provides declarations for utility functions.

/**
 * @brief Computes the number of blocks required for a given number of threads.
 *
 * @param lx The total number of threads.
 * @param thl The number of threads per block.
 * @return The number of blocks required.
 */
unsigned int blkn(unsigned int lx, int thl){
    return (lx + thl - 1) / thl; 
}

/**
 * @brief Computes the block dimensions for a 1D grid.
 *
 * @param dx The total number of threads in the x-dimension.
 * @param thrx The number of threads per block in the x-dimension.
 * @return A dim3 object representing the block dimensions.
 */
const dim3 blk_(unsigned int dx, int thrx){
    return dim3(blkn(dx, thrx)); 
}

/**
 * @brief Computes the block dimensions for a 2D grid.
 *
 * @param dx The total number of threads in the x-dimension.
 * @param thrx The number of threads per block in the x-dimension.
 * @param dy The total number of threads in the y-dimension.
 * @param thry The number of threads per block in the y-dimension.
 * @return A dim3 object representing the block dimensions.
 */
const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry){
    return dim3(blkn(dx, thrx), blkn(dy, thry)); 
}

/**
 * @brief Computes the block dimensions for a 3D grid.
 *
 * @param dx The total number of threads in the x-dimension.
 * @param thrx The number of threads per block in the x-dimension.
 * @param dy The total number of threads in the y-dimension.
 * @param thry The number of threads per block in the y-dimension.
 * @param dz The total number of threads in the z-dimension.
 * @param thrz The number of threads per block in the z-dimension.
 * @return A dim3 object representing the block dimensions.
 */
const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry, unsigned int dz, int thrz){
    return dim3(blkn(dx, thrx), blkn(dy, thry), blkn(dz, thrz)); 
}

/**
 * @brief Splits a string into a vector of substrings based on a delimiter.
 *
 * @param inpt The input string to split.
 * @param search The delimiter string.
 * @return A vector of substrings.
 */
std::vector<std::string> split(std::string inpt, std::string search) {
    size_t pos = 0;
    size_t s_dim = search.length();
    size_t index = 0;
    std::string token;
    std::vector<std::string> out = {};
    while ((pos = inpt.find(search)) != std::string::npos){
        out.push_back(inpt.substr(0, pos));
        inpt.erase(0, pos + s_dim);
        ++index;
    }
    out.push_back(inpt);
    return out;
}

/**
 * @brief Changes the device of a tensor to the specified device.
 *
 * @param dev The device string (e.g., "cuda:0").
 * @param inx Pointer to the input tensor.
 * @return A tensor on the specified device.
 */
torch::Tensor changedev(std::string dev, torch::Tensor* inx){
    c10::DeviceType dev_enm = c10::kCUDA;  
    int dev_num = 0; 
    std::vector<std::string> dex = split(dev, ":"); 
    if (dex.size() > 0){dev_num = std::stoi(dex[1]);}
    torch::TensorOptions op = torch::TensorOptions(dev_enm, dev_num); 
    return inx -> to(op.device(), false);
}

/**
 * @brief Sets the current CUDA device to the device of the input tensor.
 *
 * @param inpt Pointer to the input tensor.
 */
void changedev(torch::Tensor* inpt){
    c10::cuda::set_device(inpt -> get_device());
}

/**
 * @brief Creates tensor options based on the properties of an input tensor.
 *
 * @param v Pointer to the input tensor.
 * @return Tensor options with the same data type and device as the input tensor.
 */
torch::TensorOptions MakeOp(torch::Tensor* v){
    return torch::TensorOptions().dtype(v -> scalar_type()).device(v -> device()); 
}

/**
 * @brief Formats a tensor to have the specified dimensions and ensures it is contiguous.
 *
 * @param inpt Pointer to the input tensor.
 * @param dim The desired dimensions.
 * @return A tensor with the specified dimensions.
 */
torch::Tensor format(torch::Tensor* inpt, std::vector<signed long> dim){
    return inpt -> view(dim).contiguous();
}

/**
 * @brief Formats a vector of tensors to have the specified dimensions and concatenates them.
 *
 * @param v A vector of tensors.
 * @param dim The desired dimensions.
 * @return A concatenated tensor with the specified dimensions.
 */
torch::Tensor format(std::vector<torch::Tensor> v, std::vector<signed long> dim){
    std::vector<torch::Tensor> out; 
    for (size_t x(0); x < v.size(); ++x){out.push_back(v[x].view(dim));}
    return torch::cat(out, -1); 
}


