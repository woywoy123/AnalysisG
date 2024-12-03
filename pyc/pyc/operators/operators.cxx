#include <operators/operators.h>
#include <cutils/utils.h>

torch::Tensor operators_::Dot(torch::Tensor* v1, torch::Tensor* v2){
    return ((*v1)*(*v2)).sum({-1}, true); 
}

torch::Tensor operators_::CosTheta(torch::Tensor* v1, torch::Tensor* v2){
    torch::Tensor v1_2 = operators_::Dot(v1, v1);
    torch::Tensor v2_2 = operators_::Dot(v2, v2);
    torch::Tensor dot  = operators_::Dot(v1, v2); 
    return dot/( torch::sqrt(v1_2 * v2_2) ); 
}

torch::Tensor operators_::SinTheta(torch::Tensor* v1, torch::Tensor* v2){    
    torch::Tensor v1_2 = operators_::Dot(v1, v1);
    torch::Tensor v2_2 = operators_::Dot(v2, v2);
    torch::Tensor dot2  = torch::pow(operators_::Dot(v1, v2), 2); 
    return torch::sqrt(1 - dot2/(v1_2*v2_2));  
}

torch::Tensor operators_::Rx(torch::Tensor* angle){	
    torch::Tensor agl = angle -> view({-1, 1});
    torch::Tensor cos = torch::cos(agl); 
    torch::Tensor sin = torch::sin(agl); 
    
    torch::Tensor t0 = torch::zeros_like(agl); 
    torch::Tensor t1 = torch::ones_like(agl);
    
    return torch::cat({t1,  t0,   t0, 
                       t0, cos, -sin, 
                       t0, sin,  cos
    }, -1).view({-1, 3, 3}); 
}

torch::Tensor operators_::Ry(torch::Tensor* angle){
    torch::Tensor agl = angle -> view({-1, 1});
    torch::Tensor cos = torch::cos(agl); 
    torch::Tensor sin = torch::sin(agl); 
    
    torch::Tensor t0 = torch::zeros_like(agl); 
    torch::Tensor t1 = torch::ones_like(agl);
    
    return torch::cat({cos, t0, sin, 
                        t0, t1,  t0, 
                      -sin, t0, cos
    }, -1).view({-1, 3, 3});
}

torch::Tensor operators_::Rz(torch::Tensor* angle){
    torch::Tensor agl = angle -> view({-1, 1});
    torch::Tensor cos = torch::cos(agl); 
    torch::Tensor sin = torch::sin(agl); 
    
    torch::Tensor t0 = torch::zeros_like(agl); 
    torch::Tensor t1 = torch::ones_like(agl);
    
    return torch::cat({cos, -sin, t0, 
                       sin,  cos, t0, 
                       t0 ,   t0, t1
    }, -1).view({-1, 3, 3}); 
}

torch::Tensor operators_::CoFactors(torch::Tensor* matrix){
    int _x[] = {1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1}; 
    int _y[] = {1, 2, 1, 2, 0, 2, 0, 2, 0, 1, 0, 1}; 
    
    std::vector<torch::Tensor> out;  
    for (int i(0); i < 3; ++i){
        for (int j(0); j < 3; ++j){
            int idx = 4*i; 
            int idy = 4*j; 
            torch::Tensor a = matrix -> index({torch::indexing::Slice(), _x[idx  ], _y[idy  ]}).view({-1, 1}); 
            torch::Tensor b = matrix -> index({torch::indexing::Slice(), _x[idx+1], _y[idy+1]}).view({-1, 1}); 
            torch::Tensor c = matrix -> index({torch::indexing::Slice(), _x[idx+2], _y[idy+2]}).view({-1, 1}); 
            torch::Tensor d = matrix -> index({torch::indexing::Slice(), _x[idx+3], _y[idy+3]}).view({-1, 1}); 
            torch::Tensor minor = a*d - b*c; 
            minor = minor*( 1 - ((i+j)%2)*2 );  
            out.push_back(minor); 
        }
    } 
    return torch::cat(out, -1).view({-1, 3, 3}); 
}

torch::Tensor operators_::Determinant(torch::Tensor* matrix){ return torch::det(*matrix); }
torch::Tensor operators_::Inverse(torch::Tensor* matrix){ return torch::inverse(*matrix); }





