#ifndef H_CONUIX_MEMORY
#define H_CONUIX_MEMORY

long double** alloc(
    const int dim_i, const int dim_j
); 

long double** flush(
    long double** dmx, const int dim_j
); 

void copy(
    long double** from, long double** to, 
    const int dim_i, const int dim_j
); 

void ops(
    long double** O, long double** A, long double** B, 
    const int dim_i, const int dim_j, 
    const float sign
); 

void opm(
    long double** O, long double** A, 
    const int dim_i, const int dim_j, 
    const float c
); 

void opt(
    long double** O, long double** A, 
    const int dim_i, const int dim_j
); 




#endif
