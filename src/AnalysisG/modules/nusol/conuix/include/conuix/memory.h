#ifndef H_CONUIX_MEMORY
#define H_CONUIX_MEMORY

double** alloc(
    const int dim_i, const int dim_j
); 

double** flush(
    double** dmx, const int dim_j
); 

void copy(
    double** from, double** to, 
    const int dim_i, const int dim_j
); 

void ops(
    double** O, double** A, double** B, 
    const int dim_i, const int dim_j, 
    const float sign
); 

void opm(
    double** O, double** A, 
    const int dim_i, const int dim_j, 
    const float c
); 

void opt(
    double** O, double** A, 
    const int dim_i, const int dim_j
); 




#endif
