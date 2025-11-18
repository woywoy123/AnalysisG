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

long double& matrix_t::at(int _r, int _c){
    if (this -> r < _r || this -> c < _c){
        std::cout << "Invalid memory access: \n"; 
        std::cout << "r_matrix " << _r << " -> " << "r_in" << _r << "\n"; 
        std::cout << "c_matrix " << _r << " -> " << "c_in" << _r << "\n"; 
        std::cout << std::endl;
        abort(); 
    }
    return this -> data[_r][_c]; 
}

const long double& matrix_t::at(int _r, int _c) const {
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

matrix_t matrix_t::operator+(const matrix_t& o) const {
    matrix_t mx(this -> r, this -> c); 
    ops(mx.data, this -> data, o.data, this -> r, this -> c, 1); 
    return mx;
}

matrix_t matrix_t::operator+(const matrix_t& o){
    matrix_t mx(this -> r, this -> c); 
    ops(mx.data, this -> data, o.data, this -> r, this -> c, 1); 
    return mx;
}

matrix_t matrix_t::operator-(const matrix_t& o) const {
    matrix_t mx(this -> r, this -> c); 
    ops(mx.data, this -> data, o.data, this -> r, this -> c, -1); 
    return mx;
}

matrix_t matrix_t::operator-(const matrix_t& o){
    matrix_t mx(this -> r, this -> c); 
    ops(mx.data, this -> data, o.data, this -> r, this -> c, -1); 
    return mx;
}


matrix_t matrix_t::operator*(long double s) const {
    matrix_t mx(this -> r, this -> c); 
    opm(mx.data, this -> data, this -> r, this -> c, s); 
    return mx;
}

matrix_t matrix_t::operator*(long double s){
    matrix_t mx(this -> r, this -> c); 
    opm(mx.data, this -> data, this -> r, this -> c, s); 
    return mx;
}

matrix_t matrix_t::cross(const matrix_t& o){
    matrix_t mx(3, 3); 
    for (int i(0); i < 3; ++i){
        mx.at(0, i) = o.at(0, 1) * this -> at(2, i) - o.at(0, 2) * this -> at(1, i); 
        mx.at(1, i) = o.at(0, 2) * this -> at(0, i) - o.at(0, 0) * this -> at(2, i); 
        mx.at(2, i) = o.at(0, 0) * this -> at(1, i) - o.at(0, 1) * this -> at(0, i); 
    }    
   return mx;
}

long double matrix_t::det(){
    return _det(this -> data);
}

matrix_t matrix_t::coef(){
    matrix_t out(this -> r, this -> c); 
    out.data[0][0] =   _m_00(this -> data); 
    out.data[1][0] = - _m_10(this -> data); 
    out.data[2][0] =   _m_20(this -> data);
    out.data[0][1] = - _m_01(this -> data); 
    out.data[1][1] =   _m_11(this -> data); 
    out.data[2][1] = - _m_21(this -> data);
    out.data[0][2] =   _m_02(this -> data); 
    out.data[1][2] = - _m_12(this -> data); 
    out.data[2][2] =   _m_22(this -> data);
    return out; 
}

matrix_t matrix_t::inv(){
    auto inv3x3 =[this]() -> matrix_t{
        long double det_ = this -> det();
        det_ = (!det_) ? 0.0 : 1.0/det_; 
        matrix_t d = this -> coef() * det_;
        return d.T();
    }; 
    if (this -> c == 3 && this -> r == 3){return inv3x3();}
    return matrix_t(this -> c, this -> r); 
}

matrix_t matrix_t::diag(long double v){
    matrix_t o(this -> c, this -> r); 
    for (int x(0); x < this -> r; ++x){o.at(x, x) = v;}
    return o; 
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


