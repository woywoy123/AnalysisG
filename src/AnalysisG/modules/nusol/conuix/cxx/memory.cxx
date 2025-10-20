#include <conuix/memory.h>

double** alloc(
    const int dim_i, const int dim_j
){
    double** out = new double*[dim_i]; 
    for (int i(0); i < dim_i; ++i){out[i] = new double[dim_j]();}
    return out; 
}

double** flush(
    double** dmx, const int dim_i
){
    for (int i(0); i < dim_i; ++i){delete [] dmx[i];} 
    delete [] dmx; 
    return nullptr; 
}

void copy(
    double** from, double** to, 
    const int dim_i, const int dim_j
){
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){to[i][j] = from[i][j];}
    }
}

void ops(
    double** O, double** A, double** B, 
    const int dim_i, const int dim_j, 
    const float sign
){ 
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){O[i][j] = A[i][j] + sign * B[i][j];}
    }
}

void opm(
    double** O, double** A, 
    const int dim_i, const int dim_j, 
    const float c
){
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){O[i][j] = c * A[i][j];}
    }
}

void opt(
    double** O, double** A, 
    const int dim_i, const int dim_j
){
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){O[j][i] = A[i][j];}
    }
}



