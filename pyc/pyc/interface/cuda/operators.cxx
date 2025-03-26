#include <pyc/cupyc.h>
#include <utils/utils.cuh>
#include <physics/physics.cuh>
#include <operators/operators.cuh>

torch::Tensor pyc::operators::Dot(torch::Tensor v1, torch::Tensor v2){
    changedev(&v1); 
    return operators_::Dot(&v1, &v2); 
}

torch::Tensor pyc::operators::CosTheta(torch::Tensor v1, torch::Tensor v2){
    changedev(&v1);
    return operators_::CosTheta(&v1, &v2); 
}

torch::Tensor pyc::operators::SinTheta(torch::Tensor v1, torch::Tensor v2){
    changedev(&v1); 
    return operators_::SinTheta(&v1, &v2); 
}

torch::Tensor pyc::operators::Rx(torch::Tensor angle){
    changedev(&angle); 
    return operators_::Rx(&angle); 
}

torch::Tensor pyc::operators::Ry(torch::Tensor angle){
    changedev(&angle); 
    return operators_::Ry(&angle); 
}

torch::Tensor pyc::operators::Rz(torch::Tensor angle){
    changedev(&angle); 
    return operators_::Rz(&angle); 
}


torch::Tensor pyc::operators::RT(torch::Tensor pmc_b, torch::Tensor pmc_mu){
    changedev(&pmc_b); 
    torch::Tensor phi = pyc::transform::combined::Phi(pmc_mu);
    torch::Tensor theta = physics_::Theta(&pmc_mu); 
    return operators_::RT(&pmc_b, &phi, &theta); 
}

torch::Tensor pyc::operators::CoFactors(torch::Tensor matrix){
    changedev(&matrix); 
    return operators_::CoFactors(&matrix); 
}

torch::Tensor pyc::operators::Determinant(torch::Tensor matrix){
    changedev(&matrix); 
    return operators_::Determinant(&matrix); 
}

std::tuple<torch::Tensor, torch::Tensor> pyc::operators::Inverse(torch::Tensor matrix){
    changedev(&matrix); 
    return operators_::Inverse(&matrix); 
}

torch::Tensor pyc::operators::Cross(torch::Tensor mat1, torch::Tensor mat2){
    changedev(&mat1); 
    return operators_::Cross(&mat1, &mat2); 
}

std::tuple<torch::Tensor, torch::Tensor> pyc::operators::Eigenvalue(torch::Tensor matrix){
    changedev(&matrix); 
    return operators_::Eigenvalue(&matrix); 
}

