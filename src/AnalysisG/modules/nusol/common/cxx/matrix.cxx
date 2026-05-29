#include <common/matrix.h>
#include <common/memory.h>
#include <iostream>
#include <iomanip>

matrix_t::matrix_t(int _r, int _c){
    this -> r = _r; 
    this -> c = _c; 
    this -> data = alloc(this -> r, this -> c); 
}

matrix_t::matrix_t(const matrix_t& o){
    this -> r = o.r; this -> c = o.c; 
    this -> data = alloc(this -> r, this -> c); 
    copy(o.data, this -> data, this -> r, this -> c); 
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

matrix_t operator*(long double s, const matrix_t& o){
    matrix_t mx(o.r, o.c); 
    opm(mx.data, o.data, o.r, o.c, s); 
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
    if (this -> c == 4 && this -> r == 4){
        long double det = 0; 
        for (int x(0); x < 4; ++x){
            int mi = 0;
            matrix_t mx(3, 3); 
            for (int i(1); i < 4; ++i){
                int mj = -1; 
                for (int j(0); j < 4; ++j){
                    if (j == x){continue;}
                    mx.at(mi, ++mj) = this -> at(i, j); 
                }
                ++mi; 
            }
            det += ( (x % 2) ? -1.0L : 1.0L ) * mx.det() * this -> at(0, x); 
        }
        return det;
    }
    if (this -> c == 3 && this -> r == 3){return _det3(this -> data);}
    if (this -> c == 2 && this -> r == 2){return _det2(this -> data);}
    return 0; 
}

matrix_t matrix_t::coef(){
    matrix_t out(this -> r, this -> c); 
    if (this -> c == 3 && this -> r == 3){
        out.data[0][0] =   _m_00(this -> data); 
        out.data[1][0] = - _m_10(this -> data); 
        out.data[2][0] =   _m_20(this -> data);
        out.data[0][1] = - _m_01(this -> data); 
        out.data[1][1] =   _m_11(this -> data); 
        out.data[2][1] = - _m_21(this -> data);
        out.data[0][2] =   _m_02(this -> data); 
        out.data[1][2] = - _m_12(this -> data); 
        out.data[2][2] =   _m_22(this -> data);
    }

    if (this -> c == 2 && this -> r == 2){
        out.data[0][0] =   this -> data[1][1]; 
        out.data[1][0] = - this -> data[1][0]; 
        out.data[0][1] = - this -> data[0][1]; 
        out.data[1][1] =   this -> data[0][0]; 
    }
    return out; 
}

matrix_t matrix_t::inv(){
    auto inv4x4 =[this]() -> matrix_t{
        long double det_ = this -> det();
        if (!det_){return matrix_t(4, 4);}
        det_ = 1.0 / det_;
        matrix_t mtx(4, 4); 
        for (int i(0); i < 4; ++i){
            for (int j(0); j < 4; ++j){
                int mi = 0; matrix_t mtm(3, 3); 
                for (int _r(0); _r < 4; ++_r){
                    if (_r == i){continue;}
                    int mj = -1;
                    for (int _c(0); _c < 4; ++_c){
                        if (_c == j){continue;}
                        mtm.at(mi, ++mj) = this -> at(_r, _c);
                    }
                    ++mi;
                }
                mtx.at(i, j) = (((i + j) % 2) ? -1.0L : 1.0L) * mtm.det();
            }
        }
        return mtx.T() * det_;
    }; 

    auto inv3x3 =[this]() -> matrix_t{
        long double det_ = this -> det();
        det_ = (!det_) ? 0.0 : 1.0/det_; 
        matrix_t d = this -> coef() * det_;
        return d.T();
    }; 
    auto inv2x2 =[this]() -> matrix_t{
        long double det_ = this -> det();
        det_ = (!det_) ? 0.0 : 1.0/det_; 
        return this -> coef() * det_;
    }; 
   
    if (this -> c == 2 && this -> r == 2){return inv2x2();}
    if (this -> c == 3 && this -> r == 3){return inv3x3();}
    if (this -> c == 4 && this -> r == 4){return inv4x4();}
    return matrix_t(this -> c, this -> r); 
}


long double matrix_t::trace(){return _trace(this -> data, this -> r);}

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


