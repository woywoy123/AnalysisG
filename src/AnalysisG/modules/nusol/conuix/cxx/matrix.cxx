#include <conuix/matrix.h>
#include <conuix/memory.h>
#include <iostream>
#include <iomanip>

matrix_t::matrix_t(int _r, int _c){
    this -> r = _r; 
    this -> c = _c; 
    this -> data = alloc(this -> r, this -> c); 
}

matrix_t::~matrix_t(){
    this -> data = flush(this -> data, this -> r); 
}

matrix_t matrix_t::dot(const matrix_t& o){
    matrix_t mx(this -> r, o.c); 
    for (int i(0); i < this -> r; ++i){
        for (int j(0); j < o.c; ++j){
            for (int k(0); k < this -> c; ++k){
                mx.data[i][j] += this -> data[i][k] * o.data[k][j];
            }
        }
    }
    return mx; 
}

double& matrix_t::at(int _r, int _c){
    if (this -> r < _r || this -> c < _c){
        std::cout << "Invalid memory access: \n"; 
        std::cout << "r_matrix " << _r << " -> " << "r_in" << _r << "\n"; 
        std::cout << "c_matrix " << _r << " -> " << "c_in" << _r << "\n"; 
        std::cout << std::endl;
        abort(); 
    }
    return this -> data[_r][_c]; 
}

const double& matrix_t::at(int _r, int _c) const {
    if (this -> r < _r || this -> c < _c){
        std::cout << "Invalid memory access: \n"; 
        std::cout << "r_matrix " << _r << " -> " << "r_in" << _r << "\n"; 
        std::cout << "c_matrix " << _r << " -> " << "c_in" << _r << "\n"; 
        std::cout << std::endl;
        abort(); 
    }
    return this -> data[_r][_c]; 
}



matrix_t matrix_t::T() const {
    matrix_t mx(this -> c, this -> r); 
    opt(mx.data, this -> data, this -> r, this -> c); 
    return mx;
}

matrix_t& matrix_t::operator=(const matrix_t& o){
    if (this == &o){return *this;}
    flush(this -> data, this -> r);
    this -> r = o.r; this -> c = o.c; 
    this -> data = alloc(this -> r, this -> c); 
    copy(o.data, this -> data, this -> r, this -> c); 
    return *this; 
}

matrix_t& matrix_t::operator+(const matrix_t& o) const {
    matrix_t mx(this -> r, this -> c); 
    ops(mx.data, this -> data, o.data, this -> r, this -> c, 1); 
    return mx;
}

matrix_t& matrix_t::operator-(const matrix_t& o) const {
    matrix_t mx(this -> r, this -> c); 
    ops(mx.data, this -> data, o.data, this -> r, this -> c, -1); 
    return mx;
}

matrix_t& matrix_t::operator*(double s) const {
    matrix_t mx(this -> r, this -> c); 
    opm(mx.data, this -> data, this -> r, this -> c, s); 
    return mx;
}

void matrix_t::print(int p){
    std::cout << "------------------------" << std::endl;
    for (int i(0); i < this -> r; ++i){
        for (int j(0); j < this -> c; ++j){
            std::cout << std::fixed << std::setprecision(p) << this -> data[i][j] << "\t"; 
        }
        std::cout << std::endl; 
    }
    std::cout << "------------------------" << std::endl;
}


