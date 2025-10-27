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



