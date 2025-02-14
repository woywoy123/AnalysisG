#ifndef CUATOMIC_H
#define CUATOMIC_H
#include <torch/torch.h>

__device__ __constant__ const unsigned int _x[12] = {1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1}; 
__device__ __constant__ const unsigned int _y[12] = {1, 2, 1, 2, 0, 2, 0, 2, 0, 1, 0, 1}; 
__device__ __constant__ double _Deriv[3][3] = {0, -1, 0, 1, 0, 0, 0, 0, 0};
__device__ __constant__ double _circl[3][3] = {1, 0, 0, 0, 1, 0, 0, 0, -1}; 

template <typename scalar_t, size_t size_x1, size_t size_y1>
__device__ scalar_t _cofactor(scalar_t (&_M)[size_x1][size_y1], const unsigned int _idy, unsigned int _idz, bool cf = true){
    unsigned int idy = 4*_idy; 
    unsigned int idz = 4*_idz; 
    double ad = _M[ _x[idy  ] ][ _y[idz  ] ] * _M[ _x[idy+3] ][ _y[idz+3] ]; 
    double bc = _M[ _x[idy+1] ][ _y[idz+1] ] * _M[ _x[idy+2] ][ _y[idz+2] ]; 
    double _f = (cf) ? pow(-1, int(_idy) + int(_idz)) : 1; 
    return (ad - bc) * _f;
}

template <typename scalar_t>
__device__ scalar_t _div(scalar_t* p){return (*p) ? 1.0/(*p) : 0.0;}

template <typename scalar_t>
__device__ scalar_t _div(scalar_t p){return (p) ? 1.0/p : 0.0;}

template <typename scalar_t>
__device__ scalar_t _p2(scalar_t* p){return (*p)*(*p);}

template <typename scalar_t>
__device__ scalar_t _clp(scalar_t p){
    double sv = 10000000000; 
    return round(sv * p)/sv; 
}

template <typename scalar_t>
__device__ scalar_t _sqrt(scalar_t* p){
    return (signbit(*p) && _clp(*p) < -0.0) ? -sqrt(abs(*p)) : sqrt(*p);
}

template <typename scalar_t>
__device__ scalar_t _sqrt(scalar_t p){
    return (signbit(p) && _clp(p) < -0.0) ? -sqrt(abs(p)) : sqrt(p);
}

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
    return M_PI - fabs(fmod(fabs(*diff),  2*M_PI) - M_PI); 
}

template <typename scalar_t>
__device__ scalar_t px_(scalar_t* _pt, scalar_t* _phi){return (*_pt) * cos(*_phi);}

template <typename scalar_t>
__device__ scalar_t py_(scalar_t* _pt, scalar_t* _phi){return (*_pt) * sin(*_phi);}

template <typename scalar_t>
__device__ scalar_t pz_(scalar_t* _pt, scalar_t* _eta){return (*_pt) * sinh(*_eta);}

template <typename scalar_t>
__device__ scalar_t pt_(scalar_t* _px, scalar_t* _py){return sqrt((*_px) * (*_px) + (*_py) * (*_py));}

template <typename scalar_t>
__device__ scalar_t eta_(scalar_t* _px, scalar_t* _py, scalar_t* _pz){
    return (*_px + *_py) ? asinh(*_pz / sqrt((*_px) * (*_px) + (*_py) * (*_py))) : 0; 
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

template <typename scalar_t, size_t size_x1, size_t size_x2>
__device__ scalar_t _dot(scalar_t (&v1)[size_x1], scalar_t (&v2)[size_x2], unsigned int dx){
    scalar_t out = 0;  
    for (size_t x(0); x < dx; ++x){out += v1[x] * v2[x];}
    return out; 
}

template <typename scalar_t, size_t size_x1, size_t size_x2>
__device__ scalar_t _dot(scalar_t (&v1)[size_x1], scalar_t (&v2)[size_x2], unsigned int ds, unsigned int de){
    scalar_t out = 0;  
    for (size_t x(ds); x < de; ++x){out += v1[x] * v2[x];}
    return out; 
}

template <typename scalar_t, size_t size_x1>
__device__ scalar_t _sum(scalar_t (&v1)[size_x1], const unsigned int dx){
    scalar_t out = 0;  
    for (size_t x(0); x < dx; ++x){out += v1[x];}
    return out; 
}

template <typename scalar_t>
__device__ scalar_t _sum(scalar_t (&v1)[], const unsigned int dx){
    scalar_t out = 0;  
    for (size_t x(0); x < dx; ++x){out += v1[x];}
    return out; 
}


template <typename scalar_t>
__device__ scalar_t foptim(scalar_t t, const unsigned int l){
    return cos(t)*(l == 0) + sin(t)*(l==1) + (l == 2);
}

template <typename scalar_t>
__device__ scalar_t trigger(bool con, scalar_t v1, scalar_t v2){
    return con * v1 + (!con)*v2;
}

template <typename scalar_t>
__device__ scalar_t trigger(const unsigned int dx, scalar_t v1, scalar_t v2, scalar_t v3){
    return (dx == 0)*v1 + (dx == 1)*v2 + (dx == 2)*v3;
}


#endif
