#include <utils/utils.cuh>

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

torch::Tensor changedev(std::string dev, torch::Tensor* inx){
    c10::DeviceType dev_enm = c10::kCUDA;  
    int dev_num = 0; 
    std::vector<std::string> dex = split(dev, ":"); 
    if (dex.size() > 0){dev_num = std::stoi(dex[1]);}
    torch::TensorOptions op = torch::TensorOptions(dev_enm, dev_num); 
    return inx -> to(op.device(), true);
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


