#include <conics/matrix.h>
#include <iostream>
#include <iomanip>
#include <math.h>

// ----------------- primitives ------------- //
void flush(double** dmx, int dim_i){
    for (int i(0); i < dim_i; ++i){delete [] dmx[i];}
    delete [] dmx; 
}

double** alloc(int dim_i, int dim_j){
    double** out = new double*[dim_i];
    for (int i(0); i < dim_j; ++i){out[i] = new double[dim_j]();}
    return out; 
}

void copy(double** from, double** to, int dim_i, int dim_j){
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){to[i][j] = from[i][j];}
    }
}

void ops_add(double** O, double** A, double** B, int dim_i, int dim_j){
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){O[i][j] = A[i][j] + B[i][j];}
    }
}

void ops_sub(double** O, double** A, double** B, int dim_i, int dim_j){
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){O[i][j] = A[i][j] - B[i][j];}
    }
}

void ops_mul(double** O, double** A, double s, int dim_i, int dim_j){
    for (int i(0); i < dim_i; ++i){
        for (int j(0); j < dim_j; ++j){O[i][j] = A[i][j] * s;}
    }
}

void ops_trs(double** O, double** A, int dim_i, int dim_j){
    for (int i(0); i < dim_i; ++i) {
        for (int j(0); j < dim_j; ++j){O[j][i] = A[i][j];}
    }
}
// ----------------------------------------------- //


matrix_t::matrix_t(int _r, int _c){
    this -> r = _r; this -> c = _c; 
    this -> data = alloc(this -> r, this -> c); 
}

matrix_t::~matrix_t(){
    flush(this -> data, this -> r); 
    this -> data = nullptr; 
}


matrix_t& matrix_t::operator=(const matrix_t& o){
    if (this == &o){return *this;}
    flush(this -> data, this -> r); 
    this -> r = o.r; this -> c = o.c;
    this -> data = alloc(this -> r, this -> c); 
    copy(o.data, this -> data, this -> r, this -> c); 
    return *this; 
}

matrix_t matrix_t::operator+(const matrix_t& o) const {
    matrix_t mx(this -> r, this -> c);
    ops_add(mx.data, this -> data, o.data, this -> r, this -> c); 
    return mx;
}

matrix_t matrix_t::operator-(const matrix_t& o) const {
    matrix_t mx(this -> r, this -> c);
    ops_sub(mx.data, this -> data, o.data, this -> r, this -> c); 
    return mx; 
}

matrix_t matrix_t::operator*(double sc) const {
    matrix_t mx(this -> r, this -> c);
    ops_mul(mx.data, this -> data, sc, this -> r, this -> c); 
    return mx;
}


matrix_t matrix_t::T() const {
    matrix_t mx(this -> c, this -> r);
    ops_trs(mx.data, this -> data, this -> r, this -> c); 
    return mx;
}

matrix_t matrix_t::dot(const matrix_t& o){
    matrix_t mx(this -> r, o.c); 
    for (int i(0); i < this -> r; ++i) {
        for (int j(0); j < o.c; ++j){
            for (int k(0); k < this -> c; ++k){
                mx.data[i][j] += this -> data[i][k] * o.data[k][j];
            }
        }
    }
    return mx; 
}

double& matrix_t::at(int _r, int _c){
    return this -> data[_r][_c];
}

const double& matrix_t::at(int _r, int _c) const {
    return this -> data[_r][_c];
}

void matrix_t::print(int p){
    std::cout << "--------------" << std::endl; 
    for (int i(0); i < this -> r; ++i) {
        for (int j(0); j < this -> c; ++j){
            std::cout << std::fixed << std::setprecision(p) << this -> data[i][j] << " \t";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------" << std::endl; 
}

