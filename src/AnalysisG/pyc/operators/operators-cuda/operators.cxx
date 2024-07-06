#include <operators/operators-cuda.h>

torch::Tensor operators::cuda::Dot(torch::Tensor v1, torch::Tensor v2){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(v1.get_device()); 
    torch::Tensor output = _Dot(v1, v2); 
    c10::cuda::set_device(current_device);
    return output;
}

torch::Tensor operators::cuda::Mul(torch::Tensor v1, torch::Tensor v2){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(v1.get_device()); 
    torch::Tensor output = _Mul(v1, v2); 
    c10::cuda::set_device(current_device);
    return output;
}

torch::Tensor operators::cuda::CosTheta(torch::Tensor v1, torch::Tensor v2, signed int limit){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(v1.get_device()); 
    torch::Tensor output = _CosTheta(v1, v2, limit); 
    c10::cuda::set_device(current_device);
    return output;
}

torch::Tensor operators::cuda::SinTheta(torch::Tensor v1, torch::Tensor v2, signed int limit){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(v1.get_device()); 
    torch::Tensor output = _SinTheta(v1, v2, limit); 
    c10::cuda::set_device(current_device);
    return output;
}

torch::Tensor operators::cuda::Rx(torch::Tensor angle){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(angle.get_device()); 
    torch::Tensor output = _Rot(angle, 0); 
    c10::cuda::set_device(current_device);
    return output;
}

torch::Tensor operators::cuda::Ry(torch::Tensor angle){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(angle.get_device()); 
    torch::Tensor output = _Rot(angle, 1); 
    c10::cuda::set_device(current_device);
    return output;
}

torch::Tensor operators::cuda::Rz(torch::Tensor angle){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(angle.get_device()); 
    torch::Tensor output = _Rot(angle, 2); 
    c10::cuda::set_device(current_device);
    return output;
}

torch::Tensor operators::cuda::CoFactors(torch::Tensor matrix){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(matrix.get_device()); 
    torch::Tensor output = _CoFactors(matrix); 
    c10::cuda::set_device(current_device);
    return output;
}

torch::Tensor operators::cuda::Determinant(torch::Tensor matrix){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(matrix.get_device());
    torch::Tensor output = _Det(matrix); 
    c10::cuda::set_device(current_device);
    return output;
}

torch::Tensor operators::cuda::Inverse(torch::Tensor matrix){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(matrix.get_device()); 
    torch::Tensor output = std::get<0>(_Inv(matrix)); 
    c10::cuda::set_device(current_device);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> operators::cuda::Inverse(torch::Tensor matrix, bool det){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(matrix.get_device()); 
    std::tuple<torch::Tensor, torch::Tensor> output = _Inv(matrix);  
    c10::cuda::set_device(current_device);
    return output;
}

torch::Tensor operators::cuda::Cross(torch::Tensor mat1, torch::Tensor mat2){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(mat1.get_device()); 
    torch::Tensor output = _Cross(mat1, mat2); 
    c10::cuda::set_device(current_device);
    return output;
}

