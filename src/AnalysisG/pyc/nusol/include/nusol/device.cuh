#ifndef CU_NUSOL_DEVICE_H
#define CU_NUSOL_DEVICE_H
#include <atomic/cuatomic.cuh>

struct nusol {
    double cos, sin;
    double x0, x0p; 
    double sx, sy;
    double w, w_; 
    double x1, y1; 
    double z, o2, eps2; 

    // index: 0 -> lepton, 1 -> b-quark
    double pmass[2]  = {0x0};
    double betas[2]  = {0x0};
    double pmu_b[4]  = {0x0};
    double pmu_l[4]  = {0x0};
    double masses[3] = {0x0}; 
    bool passed = true; 
    nusol() = default; 
}; 

__device__ __forceinline__ void _makeNuSol(nusol* sl){
    double b2l = sl -> betas[0]; 

    for (size_t x(0); x < 3; ++x){sl -> masses[x] = pow(sl -> masses[x], 2);}
    for (size_t x(0); x < 2; ++x){sl -> betas[x] = _sqrt(&sl -> betas[x]);}
    sl -> eps2 = (sl -> masses[1] - sl -> masses[2])*(1 - b2l); 

    sl -> sin = 1 - pow(sl -> cos, 2); 
    sl -> sin = _sqrt(&sl -> sin); 
    double div_sin = _div(&sl -> sin); 

    double r = sl -> betas[0] * _div(&sl -> betas[1]); 
    sl -> w    = ( r - sl -> cos) * div_sin; 
    sl -> w_   = (-r - sl -> cos) * div_sin; 
    sl -> o2   = pow(sl -> w, 2) + 1 - b2l; 

    sl -> x0  = - (sl -> masses[1] - sl -> masses[2] - sl -> pmass[0]) * _div(&sl -> pmu_l[3]) * 0.5;
    sl -> x0p = - (sl -> masses[0] - sl -> masses[1] - sl -> pmass[1]) * _div(&sl -> pmu_b[3]) * 0.5;

    sl -> sx = (sl -> x0 * sl -> betas[0] - sl -> betas[0] * sl -> pmu_l[3] * (1 - b2l)) * _div(&b2l); 
    sl -> sy = (sl -> x0p * _div(&sl -> betas[1]) - sl -> cos * sl -> sx) * div_sin; 
 
    double _div_o2 = _div(&sl -> o2); 
    sl -> x1 = sl -> sx - (sl -> sx + sl -> w * sl -> sy) * _div_o2; 
    sl -> y1 = sl -> sy - (sl -> sx + sl -> w * sl -> sy) * sl -> w * _div_o2; 
    sl -> passed *= (_div_o2 > 0) * (r > 0); 

    double z2 = pow(sl -> x1, 2)*sl -> o2; 
    z2 -= pow(sl -> sy - sl -> w * sl -> sx, 2); 
    z2 -= (sl -> masses[1] - pow(sl -> x0, 2) - sl -> eps2); 
    sl -> z = (z2 <= 0) ? 0 : _sqrt(&z2); 
}


__device__ __forceinline__ double _krot(nusol* sl, const unsigned int _iky, const unsigned int _ikz){
    double r = ((_ikz < 2)*(_iky < 2) + (_ikz == _iky)) > 0; 
    r *= (_ikz == _iky)*sl -> cos + (_iky - _ikz)*sl -> sin;
    return r; 
}

__device__ __forceinline__ double _amu(nusol* sl, const unsigned int _iky, const unsigned int _ikz){
    bool ii = (_ikz == _iky); 

    double b2 = pow(sl -> betas[0], 2);
    double val = ii*(_iky == 0)*(1 - b2);
    val += ii*(_iky == 3)*(sl -> masses[1] - pow(sl -> x0, 2) - sl -> eps2); 
    val += ii*(_ikz == 2 + _ikz == 1); 
    val += ((_ikz == 3)*(_iky == 0) + (_ikz == 0)*(_iky == 3))*(sl -> sx * b2); 
    return val; 
}

__device__ __forceinline__ double _abq(nusol* sl, const unsigned int _iky, const unsigned int _ikz){
    bool ii = (_ikz == _iky); 

    double val = ii*(_iky == 0)*(1 - pow(sl -> betas[1], 2));
    val += ii*(_iky == 3)*(sl -> masses[1] - pow(sl -> x0p, 2)); 
    val += ii*(_ikz == 2 + _ikz == 1); 
    val += ((_ikz == 3)*(_iky == 0) + (_ikz == 0)*(_iky == 3))*(sl -> betas[1] * sl -> x0p); 
    return val; 
}


// ---- Htilde -----
// [Z/o, 0, x1 - P_l], [w * Z / o, 0, y1], [0, Z, 0]
__device__ __forceinline__ double _htilde(nusol* sl, const unsigned int _iky, const unsigned int _ikz){
    double z_div_o = _sqrt( &sl -> o2 ); 
    z_div_o = (sl -> z) * _div(&z_div_o); 

    double val = (_ikz == _iky) * (_ikz == 0) * z_div_o;  // Z / o
    val += (_iky == 1) * (_ikz == 0) * z_div_o * sl -> w; // w * Z / o
    val += (_iky == 2) * (_ikz == 1) * sl -> z;           // Z
    val += (_iky == 0) * (_ikz == 2) * (sl -> x1 - sl -> betas[0] * sl -> pmu_l[3]); // x1 - P_l
    val += (_iky == 1) * (_ikz == 2) * (sl -> y1);
    return val;  
}


__device__ __forceinline__ double _case1(double G[3][3], const unsigned int _idy, const unsigned int _idz){
    if (_idy == 0 && _idz == 0){return G[0][1];}
    if (_idy == 0 && _idz == 2){return G[1][2];}
    if (_idy == 1 && _idz == 1){return G[0][1];}
    if (_idy == 1 && _idz == 2){return G[0][2] - G[1][2];}
    return 0; 
}

__device__ __forceinline__ double _case2(double G[3][3], const unsigned int _idy, const unsigned int _idz, bool swpXY){
    if (!swpXY){return G[_idy][_idz];}
    if (_idy == 0 && _idz == 0){return G[1][1];}
    if (_idy == 0 && _idz == 1){return G[1][0];}
    if (_idy == 0 && _idz == 2){return G[1][2];}

    if (_idy == 1 && _idz == 0){return G[0][1];}
    if (_idy == 1 && _idz == 1){return G[0][0];}
    if (_idy == 1 && _idz == 2){return G[0][2];}

    if (_idy == 2 && _idz == 0){return G[2][1];}
    if (_idy == 2 && _idz == 1){return G[2][0];}
    if (_idy == 2 && _idz == 2){return G[2][2];}
    return 0; 
}

__device__ __forceinline__ double _leqnulls(double coF[3][3], double Q[3][3], const unsigned int _idy, const unsigned int _idz){
    if (_idy >= 2){return 0;}
    double q00 = -coF[0][0]; 
    if (q00 < 0){return 0;}

    if (_idy == 0 && _idz == 0){return Q[0][1];}
    if (_idy == 0 && _idz == 1){return Q[1][1];}
    if (_idy == 0 && _idz == 2){return Q[1][2] - _sqrt(&q00);}
    if (q00 == 0){return 0;}

    if (_idy == 1 && _idz == 0){return Q[0][1];}
    if (_idy == 1 && _idz == 1){return Q[1][1];}
    if (_idy == 1 && _idz == 2){return Q[1][2] + _sqrt(&q00);}
    return 0; 
}

__device__ __forceinline__ double _gnulls(double coF[3][3], double Q[3][3], const unsigned int _idy, const unsigned int _idz){
    if (_idy >= 2){return 0;}
    double s22 = _sqrt(-coF[2][2]);
    double q22 = _div(coF[2][2]); 

    if (_idy == 0 && _idz == 0){return  Q[0][1] - s22;}
    if (_idy == 0 && _idz == 1){return  Q[1][1];}
    if (_idy == 0 && _idz == 2){return -Q[1][1] * coF[1][2] * q22 - (Q[0][1] - s22) * coF[0][2] * q22;}

    if (_idy == 1 && _idz == 0){return  Q[0][1] + s22;}
    if (_idy == 1 && _idz == 1){return  Q[1][1];}
    if (_idy == 1 && _idz == 2){return -Q[1][1] * coF[1][2] * q22 - (Q[0][1] + s22) * coF[0][2] * q22;}
    return 0; 
}

#endif
