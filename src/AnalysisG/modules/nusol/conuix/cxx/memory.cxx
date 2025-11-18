#include <conuix/memory.h>

long double** alloc(
    const int dim_i, const int dim_j
){
    long double** out = new long double*[dim_i]; 
    for (int i(0); i < dim_i; ++i){out[i] = new long double[dim_j]();}
    return out; 
}

long double** flush(
    long double** dmx, const int dim_i
){
    for (int i(0); i < dim_i; ++i){delete [] dmx[i];} 
    delete [] dmx; 
    return nullptr; 
}

void copy(
    long double** from, long double** to, 
    const int dim_i, const int dim_j
){
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){to[i][j] = from[i][j];}
    }
}

void ops(
    long double** O, long double** A, long double** B, 
    const int dim_i, const int dim_j, 
    const float sign
){ 
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){O[i][j] = A[i][j] + sign * B[i][j];}
    }
}

void opm(
    long double** O, long double** A, 
    const int dim_i, const int dim_j, 
    const float c
){
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){O[i][j] = c * A[i][j];}
    }
}

void opt(
    long double** O, long double** A, 
    const int dim_i, const int dim_j
){
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){O[j][i] = A[i][j];}
    }
}


long double _trace(long double** A){return A[0][0] + A[1][1] + A[2][2];}
long double _m_00(long double**  M){return M[1][1] * M[2][2] - M[1][2] * M[2][1];}
long double _m_01(long double**  M){return M[1][0] * M[2][2] - M[1][2] * M[2][0];}
long double _m_02(long double**  M){return M[1][0] * M[2][1] - M[1][1] * M[2][0];}
long double _m_10(long double**  M){return M[0][1] * M[2][2] - M[0][2] * M[2][1];}
long double _m_11(long double**  M){return M[0][0] * M[2][2] - M[0][2] * M[2][0];}
long double _m_12(long double**  M){return M[0][0] * M[2][1] - M[0][1] * M[2][0];}
long double _m_20(long double**  M){return M[0][1] * M[1][2] - M[0][2] * M[1][1];}
long double _m_21(long double**  M){return M[0][0] * M[1][2] - M[0][2] * M[1][0];}
long double _m_22(long double**  M){return M[0][0] * M[1][1] - M[0][1] * M[1][0];}
long double _det( long double**  v){return v[0][0] * _m_00(v) - v[0][1] * _m_01(v) + v[0][2] * _m_02(v);}

