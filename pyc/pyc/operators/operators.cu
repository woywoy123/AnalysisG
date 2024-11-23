#include <operators/operators.cuh>
#include <operators/base.cuh>
#include <cutils/utils.cuh>

torch::Tensor operators_::Dot(torch::Tensor* v1, torch::Tensor* v2){
    const unsigned int dx = v1 -> size({0}); 
    const unsigned int dy = v1 -> size({1}); 

    const dim3 thd = dim3(1, dy); 
    const dim3 blk = blk_(dx, 1, dy, dy); 
    torch::Tensor out = torch::zeros({dx}, MakeOp(v1)); 

    unsigned int sx = sizeof(double)*dy; 
    AT_DISPATCH_FLOATING_TYPES(v1 -> scalar_type(), "dot", [&]{
        _dot<scalar_t><<<blk, thd, sx>>>(
                v1 -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                v2 -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
                dx, dy); 
    }); 
    return out;
}

torch::Tensor operators_::Cross(torch::Tensor* v1, torch::Tensor* v2){
    const unsigned int dx = v1 -> size({0}); 
    const unsigned int dy = v1 -> size({1}); 
    const unsigned int dz = v1 -> size({2}); 

    const dim3 thd = dim3(1, dy, dz); 
    const dim3 blk = blk_(dx, 1, dy, dy, dz, dz); 
    torch::Tensor out = torch::zeros({dx, dy, dz, dz}, MakeOp(v1)); 

    unsigned int sx = sizeof(double)*dy*dy*dz*dz; 
    AT_DISPATCH_FLOATING_TYPES(v1 -> scalar_type(), "cross", [&]{
        _cross<scalar_t><<<blk, thd, sx>>>(
                v1 -> packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(), 
                v2 -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                out.packed_accessor64<scalar_t  , 4, torch::RestrictPtrTraits>(),
                dy, dz); 
    }); 
    return out;
}

torch::Tensor operators_::CosTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm){
    const unsigned int dy = (lm) ? lm : v1 -> size({1}); 
    const unsigned int dx = v1 -> size({0});
    if (dy < 3){
        torch::Tensor xx = ((*v1)*(*v1)).sum(-1, true); 
        torch::Tensor xy = ((*v1)*(*v2)).sum(-1, true);
        torch::Tensor yy = ((*v2)*(*v2)).sum(-1, true);
        return xy/torch::sqrt(xx*yy);
    } 

    const dim3 thd = dim3(1, dy);
    const dim3 blk = blk_(dx, 1, dy, dy); 
    torch::Tensor out = torch::zeros({dx, 1}, MakeOp(v1));
  
    unsigned int sx = sizeof(double)*dy*2; 
    AT_DISPATCH_ALL_TYPES(v1 -> scalar_type(), "costheta", [&]{
        _costheta<scalar_t><<<blk, thd, sx>>>(
                v1 -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                v2 -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                dx, dy); 
    }); 
    return out; 
}

torch::Tensor operators_::SinTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm){
    const unsigned int dy = (lm) ? lm : v1 -> size({1}); 
    const unsigned int dx = v1 -> size({0});
    if (dy < 3){
        torch::Tensor xx = ((*v1)*(*v1)).sum(-1, true); 
        torch::Tensor xy = ((*v1)*(*v2)).sum(-1, true);
        torch::Tensor yy = ((*v2)*(*v2)).sum(-1, true);
        return torch::sqrt(1 - torch::pow(xy/torch::sqrt(xx*yy), 2));
    } 

    const dim3 thd = dim3(1, dy);
    const dim3 blk = blk_(dx, 1, dy, dy); 
    torch::Tensor out = torch::zeros({dx, 1}, MakeOp(v1));
  
    unsigned int sx = sizeof(double)*dy*2; 
    AT_DISPATCH_ALL_TYPES(v1 -> scalar_type(), "costheta", [&]{
        _costheta<scalar_t><<<blk, thd, sx>>>(
                v1 -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                v2 -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                dx, dy, true); 
    }); 
    return out; 
}

torch::Tensor operators_::Rx(torch::Tensor* angle){
    const unsigned int dx = angle -> size({0}); 
    const dim3 thd = dim3(64, 3, 3); 
    const dim3 blk = blk_(dx, 64, 3, 3, 3, 3);  
    torch::Tensor out = torch::zeros({dx, 3, 3}, MakeOp(angle));
    AT_DISPATCH_ALL_TYPES(angle -> scalar_type(), "Rx", [&]{
        _rx<scalar_t><<<blk, thd>>>(
                angle -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                dx); 
    }); 
    return out; 
}

torch::Tensor operators_::Ry(torch::Tensor* angle){
    const unsigned int dx = angle -> size({0}); 
    const dim3 thd = dim3(64, 3, 3); 
    const dim3 blk = blk_(dx, 64, 3, 3, 3, 3);  
    torch::Tensor out = torch::zeros({dx, 3, 3}, MakeOp(angle));
    AT_DISPATCH_ALL_TYPES(angle -> scalar_type(), "Ry", [&]{
        _ry<scalar_t><<<blk, thd>>>(
                angle -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                dx); 
    }); 
    return out; 
}

torch::Tensor operators_::Rz(torch::Tensor* angle){
    const unsigned int dx = angle -> size({0}); 
    const dim3 thd = dim3(64, 3, 3); 
    const dim3 blk = blk_(dx, 64, 3, 3, 3, 3);  
    torch::Tensor out = torch::zeros({dx, 3, 3}, MakeOp(angle));
    AT_DISPATCH_ALL_TYPES(angle -> scalar_type(), "Rz", [&]{
        _rz<scalar_t><<<blk, thd>>>(
                angle -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                dx); 
    }); 
    return out; 
}

torch::Tensor operators_::RT(torch::Tensor* pmc, torch::Tensor* phi, torch::Tensor* theta){
    const unsigned int dx = pmc -> size({0}); 
    const dim3 thd = dim3(1, 3, 3); 
    const dim3 blk = blk_(dx, 1, 3, 3, 3, 3); 
    torch::Tensor out = torch::zeros({dx, 3, 3}, MakeOp(theta));
    AT_DISPATCH_ALL_TYPES(theta -> scalar_type(), "RT", [&]{
        _rt<scalar_t><<<blk, thd>>>(
                pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                phi -> packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                theta -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); 
    }); 
    return out; 
}

torch::Tensor operators_::CoFactors(torch::Tensor* matrix){
    const unsigned int dx = matrix -> size({0}); 
    const dim3 thd = dim3(1, 3, 3); 
    const dim3 blk = blk_(dx, 1, 3, 3, 3, 3); 
    torch::Tensor out = torch::zeros({dx, 3, 3}, MakeOp(matrix));

    AT_DISPATCH_ALL_TYPES(matrix -> scalar_type(), "CoFactors", [&]{
        _cofactor<scalar_t><<<blk, thd>>>(
            matrix -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); 
    }); 
    return out;
}

torch::Tensor operators_::Determinant(torch::Tensor* matrix){
    const unsigned int dx = matrix -> size({0}); 
    const dim3 thd = dim3(1, 3, 3); 
    const dim3 blk = blk_(dx, 1, 3, 3, 3, 3); 
    torch::Tensor out = torch::zeros({dx, 1}, MakeOp(matrix));
    AT_DISPATCH_ALL_TYPES(matrix -> scalar_type(), "Determinant", [&]{
        _determinant<scalar_t><<<blk, thd>>>(
            matrix -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()); 
    }); 
    return out;
}


std::tuple<torch::Tensor, torch::Tensor> operators_::Inverse(torch::Tensor* matrix){
    const unsigned int dx = matrix -> size({0}); 
    const dim3 thd = dim3(1, 3, 3); 
    const dim3 blk = blk_(dx, 1, 3, 3, 3, 3); 
    torch::Tensor det = torch::zeros({dx, 1}, MakeOp(matrix));
    torch::Tensor inv = torch::zeros({dx, 3, 3}, MakeOp(matrix)); 
    AT_DISPATCH_ALL_TYPES(matrix -> scalar_type(), "Inverse", [&]{
        _inverse<scalar_t><<<blk, thd>>>(
            matrix -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            inv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            det.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()); 
    }); 
    return {inv, det};
}

std::tuple<torch::Tensor, torch::Tensor> operators_::Eigenvalue(torch::Tensor* matrix){
    const unsigned int dx = matrix -> size({0}); 
    const dim3 thd = dim3(1, 3, 3); 
    const dim3 blk = blk_(dx, 1, 3, 3, 3, 3); 
    torch::Tensor eig = torch::zeros({dx, 3}, MakeOp(matrix)); 
    torch::Tensor img = torch::zeros({dx, 3}, MakeOp(matrix)); 
    AT_DISPATCH_ALL_TYPES(matrix -> scalar_type(), "Eigenvalue", [&]{
        _eigenvalue<scalar_t><<<blk, thd>>>(
            matrix -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            eig.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            img.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()); 
    }); 
    return {eig, img};
}




