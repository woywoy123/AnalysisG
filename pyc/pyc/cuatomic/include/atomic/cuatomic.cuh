#ifndef CUATOMIC_H
#define CUATOMIC_H
#include <torch/torch.h>

template <typename scalar_t>
__device__ scalar_t _div(scalar_t* p){return (*p) ? 1/(*p) : 0;}

template <typename scalar_t>
__device__ scalar_t _p2(scalar_t* p){return pow(*p, 2);}

template <typename scalar_t>
__device__ scalar_t _sqrt(scalar_t* p){return (1 - (*p <= 0)*2) * pow(abs(*p), 0.5);}


template <typename scalar_t>
__device__ scalar_t _cmp(scalar_t xx, scalar_t yy, scalar_t xy){
    scalar_t t = xx * yy;
    t = _sqrt(&t);  
    return xy * _div(&t); 
}

template <typename scalar_t>
__device__ scalar_t _arccos(scalar_t* sm, scalar_t* _pz){
    scalar_t data = _sqrt(sm); 
    return acos(_div(&data) * (*_pz));
}

template <typename scalar_t> 
__device__ scalar_t minus_mod(scalar_t* diff){
//    if (*diff > M_PI){return *diff - 2.0*M_PI;}
//    if (*diff <= -M_PI){return *diff + 2.0*M_PI;} 
//    return *diff; 
    return M_PI - fabs(fmod(fabs(*diff),  2*M_PI) - M_PI); 
}

template <typename scalar_t>
__device__ scalar_t px_(scalar_t* _pt, scalar_t* _phi){return (*_pt) * cos(*_phi);}

template <typename scalar_t>
__device__ scalar_t py_(scalar_t* _pt, scalar_t* _phi){return (*_pt) * sin(*_phi);}

template <typename scalar_t>
__device__ scalar_t pz_(scalar_t* _pt, scalar_t* _eta){return (*_pt) * sinh(*_eta);}

template <typename scalar_t>
__device__ scalar_t pt_(scalar_t* _px, scalar_t* _py){return sqrt(pow(*_px, 2) + pow(*_py, 2));}

template <typename scalar_t>
__device__ scalar_t eta_(scalar_t* _px, scalar_t* _py, scalar_t* _pz){
    return (*_px + *_py) ? asinh(*_pz / sqrt(pow(*_px, 2) + pow(*_py, 2))) : 0; 
}

template <typename scalar_t>
__device__ scalar_t eta_(scalar_t* _pt, scalar_t* _pz){return (*_pt) ? asinh(*_pz / *_pt) : 0;}

template <typename scalar_t>
__device__ scalar_t phi_(scalar_t* _px, scalar_t* _py){return (*_px) ? atan2(*_py, *_px) : 0;}

template <typename scalar_t>
__device__ scalar_t _rx(scalar_t* _a, const unsigned int _idy, const unsigned int _idz){
    bool lz = (_idz >= 1); 
    bool ly = (_idy >= 1); 
    bool ii = (_idy == _idz);
    scalar_t val = (!lz + lz*cos(*_a))*ii; 
    return val + (!ii)*ly*lz*(1 - 2*(_idz > _idy))*sin(*_a); 
}

template <typename scalar_t>
__device__ scalar_t _ry(scalar_t* _a, const unsigned int _idy, const unsigned int _idz){
    bool lz = (_idz != 1)*(_idy != 1); 
    bool ii = _idy == _idz; 

    scalar_t ng  = (1 - 2*(_idy > _idz)) * sin(*_a); 
    scalar_t val = ii*(!lz + lz * cos(*_a));
    return val + ng * ((_idy == 2)*(_idz == 0) + (_idy == 0)*(_idz == 2));  
}

template <typename scalar_t>
__device__ scalar_t _rz(scalar_t* _a, const unsigned int _idy, const unsigned int _idz){
    bool lz = (_idz <= 1); 
    bool ly = (_idy <= 1); 
    bool ii = (_idy == _idz);
    scalar_t val = (!lz + lz*cos(*_a))*ii; 
    return val + (!ii)*ly*lz*(1 - 2*(_idz > _idy))*sin(*_a); 
}

template <typename scalar_t, size_t size_x1, size_t size_y1, size_t size_x2, size_t size_y2>
__device__ scalar_t _dot(
        scalar_t (&v1)[size_x1][size_y1], scalar_t (&v2)[size_x2][size_y2], 
        const unsigned int row, unsigned int col, unsigned int dx
){
    scalar_t out = 0;  
    for (size_t x(0); x < dx; ++x){out += v1[row][x] * v2[x][col];}
    return out; 
}


#endif
