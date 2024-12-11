#include <cutils/utils.cuh>

unsigned int blkn(unsigned int lx, int thl){
    return (lx + thl - 1) / thl; 
}

const dim3 blk_(unsigned int dx, int thrx){
    return dim3(blkn(dx, thrx)); 
}

const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry){
    return dim3(blkn(dx, thrx), blkn(dy, thry)); 
}

const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry, unsigned int dz, int thrz){
    return dim3(blkn(dx, thrx), blkn(dy, thry), blkn(dz, thrz)); 
}

torch::Tensor changedev(std::string dev, torch::Tensor* inx){
    return inx -> to(dev);
}

void changedev(torch::Tensor* inpt){
    c10::cuda::set_device(inpt -> get_device());
}

torch::TensorOptions MakeOp(torch::Tensor* v){
    return torch::TensorOptions().dtype(v -> scalar_type()).device(v -> device()); 
}

torch::Tensor format(torch::Tensor* inpt, std::vector<signed long> dim){
    return inpt -> view(dim).contiguous();
}

torch::Tensor format(std::vector<torch::Tensor> v, std::vector<signed long> dim){
    std::vector<torch::Tensor> out; 
    for (size_t x(0); x < v.size(); ++x){out.push_back(v[x].view(dim));}
    return torch::cat(out, -1); 
}

